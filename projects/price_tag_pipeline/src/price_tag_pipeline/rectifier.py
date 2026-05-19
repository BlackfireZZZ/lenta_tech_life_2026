"""Crop rectifier.

Two paths now:
- `TagRectifier` — fast axis-aligned crop + optional 90° rotation + CLAHE on
  luminance. Default for the runtime profiles. Keeps color (yellow/red promo
  regions carry signal).
- `PerspectiveRectifier` — heuristic 4-corner detection inside the bbox via
  contour/quadrilateral search, then homography warp to a canonical
  rectangle. When it works, OCR sees a properly rectified tag instead of a
  skewed parallelogram. When it can't find a clean quad it falls back to
  the axis-aligned path.

A future P1 upgrade is a small keypoint head trained to predict the four
corners directly; the API surface here is set up to swap that in.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from .config import RectifierConfig
from .quality import laplacian_sharpness
from .types import CropCandidate, Detection

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Axis-aligned (existing fast path)
# ---------------------------------------------------------------------------

def _upright_crop(
    crop: np.ndarray, frame_rotation: str, rotate_vertical_tags: bool
) -> np.ndarray:
    """Rotate a raw-frame crop so its text is upright before OCR/decoding.

    The Lenta scan-robot cam is mounted **90° CW**, so the detector runs on
    an upright-rotated frame but un-projects boxes back to the *original*
    (sideways) frame — and the pipeline crops from that original frame
    (detector.py contract). So every crop the rectifier gets in a rotated
    profile is itself sideways: a wide tag becomes tall and, crucially, the
    text is 90° off. It must be un-rotated by the **inverse of the camera
    mounting**, exactly as the detector un-rotates the whole frame and
    `make_shelf_audit_fixture` un-rotates display crops:

      * ``ccw`` (the Lenta profiles, cam 90° CW) → ``ROTATE_90_COUNTERCLOCKWISE``
      * ``cw``                                   → ``ROTATE_90_CLOCKWISE``

    Aspect ratio is NOT the gate here — every raw crop is sideways regardless
    of how square it looks; ``rotate_vertical_tags`` is only the master
    on/off switch. ``none`` keeps the legacy "obviously-tall → 90° CW"
    heuristic so already-upright input is left untouched (the documented
    detector invariant). A previous version rotated CW in *every* case, which
    in the shipped ``ccw`` profile fed Qwen-VL crops upside-down (180° off) —
    a severe, silent OCR-quality loss.
    """
    r = (frame_rotation or "none").lower()
    if r == "ccw":
        return cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE) if rotate_vertical_tags else crop
    if r == "cw":
        return cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE) if rotate_vertical_tags else crop
    # 'none': legacy behaviour — only flip an obviously-vertical crop.
    if rotate_vertical_tags and crop.shape[0] > crop.shape[1] * 1.6:
        return cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    return crop


class TagRectifier:
    """Padded crop + optional rotation + CLAHE on luminance. Keeps color."""

    def __init__(self, cfg: RectifierConfig, frame_rotation: str = "none"):
        self.cfg = cfg
        self.frame_rotation = frame_rotation
        self._clahe = cv2.createCLAHE(
            clipLimit=self.cfg.clahe_clip_limit,
            tileGridSize=(self.cfg.clahe_tile_grid_size, self.cfg.clahe_tile_grid_size),
        )

    def rectify(self, frame: np.ndarray, det: Detection) -> CropCandidate | None:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = det.bbox_xyxy
        pad_x = int((x2 - x1) * self.cfg.padding_ratio)
        pad_y = int((y2 - y1) * self.cfg.padding_ratio)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w - 1, x2 + pad_x)
        y2 = min(h - 1, y2 + pad_y)
        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2].copy()
        if crop.size == 0:
            return None

        crop = _upright_crop(crop, self.frame_rotation, self.cfg.rotate_vertical_tags)

        enhanced, gray = _apply_clahe(crop, self._clahe, keep_color=self.cfg.keep_color)
        sharpness = laplacian_sharpness(gray)
        area_px = int(enhanced.shape[0] * enhanced.shape[1])
        return CropCandidate(
            image=enhanced,
            sharpness=sharpness,
            area_px=area_px,
            bbox_xyxy=(x1, y1, x2, y2),
            detection_confidence=det.confidence,
            frame_idx=det.frame_idx,
            timestamp_s=det.timestamp_s,
        )


# ---------------------------------------------------------------------------
# Perspective rectifier (heuristic)
# ---------------------------------------------------------------------------

class PerspectiveRectifier:
    """Try to recover a 4-point quadrilateral inside the bbox and warp it to a rectangle.

    Pipeline:
      1. Pad-crop around the detection (same as TagRectifier).
      2. Convert to grayscale, light blur, adaptive threshold.
      3. Find external contours, keep those approximating a 4-vertex polygon
         with area >= `min_quad_area_ratio` * crop area.
      4. Pick the largest such quad. Order its corners (TL, TR, BR, BL).
      5. Compute a target rectangle (aspect ratio preserved); warp via
         cv2.getPerspectiveTransform + warpPerspective.
      6. CLAHE on luminance; return CropCandidate.

    When no clean quad is found, fall back to `TagRectifier`.

    Pure OpenCV — no model dependencies. Good for tags with a clear printed
    border. For shelves where the border is obscured, the keypoint-head
    approach (P1 in docs/strategy.md) is the right upgrade.
    """

    def __init__(
        self,
        cfg: RectifierConfig,
        min_quad_area_ratio: float = 0.35,
        frame_rotation: str = "none",
    ):
        self.cfg = cfg
        self.frame_rotation = frame_rotation
        self.min_quad_area_ratio = min_quad_area_ratio
        self._fallback = TagRectifier(cfg, frame_rotation=frame_rotation)
        self._clahe = cv2.createCLAHE(
            clipLimit=cfg.clahe_clip_limit,
            tileGridSize=(cfg.clahe_tile_grid_size, cfg.clahe_tile_grid_size),
        )

    def rectify(self, frame: np.ndarray, det: Detection) -> CropCandidate | None:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = det.bbox_xyxy
        pad_x = int((x2 - x1) * self.cfg.padding_ratio)
        pad_y = int((y2 - y1) * self.cfg.padding_ratio)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w - 1, x2 + pad_x)
        y2 = min(h - 1, y2 + pad_y)
        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2].copy()
        if crop.size == 0:
            return None

        quad = _find_tag_quadrilateral(crop, self.min_quad_area_ratio)
        if quad is None:
            return self._fallback.rectify(frame, det)

        warped = _warp_to_rect(crop, quad)
        if warped is None or warped.size == 0:
            return self._fallback.rectify(frame, det)

        warped = _upright_crop(warped, self.frame_rotation, self.cfg.rotate_vertical_tags)

        enhanced, gray = _apply_clahe(warped, self._clahe, keep_color=self.cfg.keep_color)
        sharpness = laplacian_sharpness(gray)
        area_px = int(enhanced.shape[0] * enhanced.shape[1])
        return CropCandidate(
            image=enhanced,
            sharpness=sharpness,
            area_px=area_px,
            bbox_xyxy=(x1, y1, x2, y2),
            detection_confidence=det.confidence,
            frame_idx=det.frame_idx,
            timestamp_s=det.timestamp_s,
        )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _apply_clahe(crop: np.ndarray, clahe, keep_color: bool) -> tuple[np.ndarray, np.ndarray]:
    if keep_color and crop.ndim == 3:
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        l_chan = clahe.apply(l_chan)
        lab = cv2.merge([l_chan, a_chan, b_chan])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        gray = clahe.apply(gray)
        enhanced = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return enhanced, gray


def _find_tag_quadrilateral(crop: np.ndarray, min_area_ratio: float) -> Optional[np.ndarray]:
    h, w = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Adaptive threshold — robust to lighting changes across the shelf.
    th = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 6
    )
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    crop_area = float(h * w)
    best_quad: Optional[np.ndarray] = None
    best_area = 0.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area / crop_area < min_area_ratio:
            continue
        peri = cv2.arcLength(cnt, closed=True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, closed=True)
        if len(approx) == 4 and cv2.isContourConvex(approx) and area > best_area:
            best_quad = approx.reshape(-1, 2).astype(np.float32)
            best_area = area
    return best_quad


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 corner points as (TL, TR, BR, BL)."""
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.stack([tl, tr, br, bl]).astype(np.float32)


def _warp_to_rect(crop: np.ndarray, quad: np.ndarray) -> Optional[np.ndarray]:
    src = _order_corners(quad)
    tl, tr, br, bl = src
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    dst_w = int(round(max(width_top, width_bottom)))
    dst_h = int(round(max(height_left, height_right)))
    if dst_w < 20 or dst_h < 20:
        return None
    dst = np.array(
        [[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(crop, M, (dst_w, dst_h), flags=cv2.INTER_CUBIC)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_rectifier(cfg: RectifierConfig, frame_rotation: str = "none"):
    """``frame_rotation`` mirrors ``DetectorConfig.frame_rotation``: the
    rectifier crops from the *original* (possibly sideways) frame, so it must
    un-rotate crops upright by the same camera-mounting inverse the detector
    applies to whole frames. Passed by ``PriceTagPipeline``; defaults to
    ``"none"`` so direct callers keep legacy behaviour."""
    if getattr(cfg, "perspective", False):
        return PerspectiveRectifier(cfg, frame_rotation=frame_rotation)
    return TagRectifier(cfg, frame_rotation=frame_rotation)
