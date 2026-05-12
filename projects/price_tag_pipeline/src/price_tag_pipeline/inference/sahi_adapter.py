"""SAHI tiled inference for small-object detection.

Price tags are *small* relative to the full frame. SAHI slices the image into
overlapping tiles, runs the detector on each tile at native scale, and merges
predictions back into the full frame. This routinely adds 3–8 mAP for small
objects without retraining.

Two backends supported:
- The official `sahi` library (preferred — well-maintained, supports ultralytics
  and many other detectors natively).
- A self-contained fallback that does the tile loop here, suitable when sahi
  cannot be installed (e.g. weird Python version conflicts).

The fallback expects a `predict_fn(image_bgr) -> (boxes_xyxy_pixel, scores, labels)`.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import numpy as np

from .wbf import weighted_box_fusion

LOGGER = logging.getLogger(__name__)

PredictFn = Callable[[np.ndarray], tuple[list[tuple[float, float, float, float]], list[float], list[int]]]


def build_sahi_predictor(
    model_path: str,
    confidence_threshold: float = 0.25,
    device: Optional[str] = None,
    model_type: str = "ultralytics",
):
    """Return a sahi.AutoDetectionModel (or raise ImportError with install hint).

    For our YOLO pipeline, `model_type='ultralytics'` works against any
    YOLO/RT-DETR checkpoint Ultralytics can load.
    """
    try:
        from sahi import AutoDetectionModel  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "SAHI not installed. Install with: pip install sahi"
        ) from e
    return AutoDetectionModel.from_pretrained(
        model_type=model_type,
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        device=device or "cpu",
    )


def sahi_predict(
    sahi_model,
    image: np.ndarray,
    slice_height: int = 800,
    slice_width: int = 800,
    overlap_ratio: float = 0.2,
    postprocess: str = "GREEDYNMM",
):
    """Run SAHI sliced prediction and return (boxes_xyxy, scores, labels)."""
    try:
        from sahi.predict import get_sliced_prediction  # type: ignore
    except ImportError as e:
        raise RuntimeError("SAHI not installed: pip install sahi") from e
    result = get_sliced_prediction(
        image=image,
        detection_model=sahi_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
        postprocess_type=postprocess,
    )
    boxes: list[tuple[float, float, float, float]] = []
    scores: list[float] = []
    labels: list[int] = []
    for pred in result.object_prediction_list:
        bbox = pred.bbox
        boxes.append((float(bbox.minx), float(bbox.miny), float(bbox.maxx), float(bbox.maxy)))
        scores.append(float(pred.score.value))
        labels.append(int(pred.category.id))
    return boxes, scores, labels


# ---------------------------------------------------------------------------
# Fallback: pure-Python tiling
# ---------------------------------------------------------------------------

def tile_and_predict(
    image: np.ndarray,
    predict_fn: PredictFn,
    tile_size: int = 800,
    overlap: float = 0.2,
    iou_threshold: float = 0.5,
) -> tuple[list[tuple[float, float, float, float]], list[float], list[int]]:
    """Self-contained tiled inference using WBF for merging.

    Slow path; prefer `sahi_predict` when the `sahi` library is installed.
    """
    h, w = image.shape[:2]
    stride = max(1, int(tile_size * (1 - overlap)))

    tiles_boxes: list[list[tuple[float, float, float, float]]] = []
    tiles_scores: list[list[float]] = []
    tiles_labels: list[list[int]] = []

    for y0 in range(0, h, stride):
        for x0 in range(0, w, stride):
            x1 = min(x0 + tile_size, w)
            y1 = min(y0 + tile_size, h)
            if x1 - x0 < tile_size // 2 or y1 - y0 < tile_size // 2:
                continue
            tile = image[y0:y1, x0:x1]
            boxes, scores, labels = predict_fn(tile)
            # Translate tile-local coords back to full image, then normalize to [0,1] for WBF.
            full = [
                ((b[0] + x0) / w, (b[1] + y0) / h, (b[2] + x0) / w, (b[3] + y0) / h)
                for b in boxes
            ]
            tiles_boxes.append(full)
            tiles_scores.append(list(scores))
            tiles_labels.append(list(labels))

    if not tiles_boxes:
        return [], [], []

    norm_boxes, scores, labels = weighted_box_fusion(
        tiles_boxes, tiles_scores, tiles_labels, iou_threshold=iou_threshold,
    )
    boxes = [(b[0] * w, b[1] * h, b[2] * w, b[3] * h) for b in norm_boxes]
    return boxes, scores, labels
