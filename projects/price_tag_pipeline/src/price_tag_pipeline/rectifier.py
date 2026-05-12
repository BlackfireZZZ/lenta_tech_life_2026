"""Crop rectifier.

For now: padded axis-aligned crop + optional 90° rotation + CLAHE on luminance
while preserving color. A 4-point perspective unwarp via a corner-keypoint head
is on the P1 list (STRATEGY.md §1, §9) and will land once we have a trained
keypoint model or annotated tag corners.

Why preserve color: yellow/red regions on Russian price tags carry the promo
signal that lets us disambiguate regular vs loyalty price. Throwing color away
(as the scaffold did) closes that door for both the parser fallback and the
VLM path.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import RectifierConfig
from .quality import laplacian_sharpness
from .types import CropCandidate, Detection


class TagRectifier:
    def __init__(self, cfg: RectifierConfig):
        self.cfg = cfg
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

        if self.cfg.rotate_vertical_tags and crop.shape[0] > crop.shape[1] * 1.6:
            crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)

        if self.cfg.keep_color and crop.ndim == 3:
            # CLAHE on luminance only; preserve color.
            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            l_chan, a_chan, b_chan = cv2.split(lab)
            l_chan = self._clahe.apply(l_chan)
            lab = cv2.merge([l_chan, a_chan, b_chan])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        else:
            # Legacy path: convert to grayscale, CLAHE, then back to BGR.
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
            gray = self._clahe.apply(gray)
            enhanced = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

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
