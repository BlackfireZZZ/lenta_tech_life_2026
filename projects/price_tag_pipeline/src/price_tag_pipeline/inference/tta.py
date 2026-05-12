"""Test-Time Augmentation (TTA) for detectors.

Runs the detector multiple times with simple augmentations, transforms each
prediction back to the original frame, and fuses them with WBF.

Augmentations applied:
- Identity (original)
- Horizontal flip
- Multi-scale (0.83x, 1.0x, 1.17x) — optional

This module is detector-agnostic: pass it a `predict_fn(image_bgr) ->
(boxes_xyxy, scores, labels)` and it handles the rest.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from .wbf import weighted_box_fusion


PredictFn = Callable[[np.ndarray], tuple[list[tuple[float, float, float, float]], list[float], list[int]]]


def run_tta(
    image: np.ndarray,
    predict_fn: PredictFn,
    scales: tuple[float, ...] = (1.0,),
    hflip: bool = True,
    iou_threshold: float = 0.55,
) -> tuple[list[tuple[float, float, float, float]], list[float], list[int]]:
    """Run the detector under TTA and fuse with WBF.

    Box coordinates are returned in the same pixel space as `image`.
    """
    h, w = image.shape[:2]

    runs_boxes: list[list[tuple[float, float, float, float]]] = []
    runs_scores: list[list[float]] = []
    runs_labels: list[list[int]] = []

    def _add(boxes_xyxy, scores, labels, flip_x: bool, scale: float):
        # Unscale to the original size.
        if scale != 1.0:
            inv = 1.0 / scale
            boxes_xyxy = [(b[0] * inv, b[1] * inv, b[2] * inv, b[3] * inv) for b in boxes_xyxy]
        if flip_x:
            boxes_xyxy = [(w - b[2], b[1], w - b[0], b[3]) for b in boxes_xyxy]
        # Normalize to [0,1] for WBF.
        norm = [(b[0] / w, b[1] / h, b[2] / w, b[3] / h) for b in boxes_xyxy]
        runs_boxes.append(norm)
        runs_scores.append(list(scores))
        runs_labels.append(list(labels))

    for s in scales:
        if s == 1.0:
            scaled = image
        else:
            scaled = cv2.resize(image, (int(w * s), int(h * s)), interpolation=cv2.INTER_LINEAR)
        boxes, scores, labels = predict_fn(scaled)
        _add(boxes, scores, labels, flip_x=False, scale=s)

        if hflip:
            flipped = scaled[:, ::-1, :]
            boxes_f, scores_f, labels_f = predict_fn(flipped)
            # Predictions are in the flipped frame's pixel space (=scaled width).
            # We pass them through `_add` with flip_x=True AND scale=s, but
            # the flipping must happen in scaled coords before the unscale.
            sh, sw = scaled.shape[:2]
            # Un-flip: in the flipped image, x is (sw - x). Convert back to scaled coords first.
            boxes_unflipped_in_scaled = [(sw - b[2], b[1], sw - b[0], b[3]) for b in boxes_f]
            _add(boxes_unflipped_in_scaled, scores_f, labels_f, flip_x=False, scale=s)

    fused_norm_boxes, fused_scores, fused_labels = weighted_box_fusion(
        runs_boxes, runs_scores, runs_labels, iou_threshold=iou_threshold,
    )
    # De-normalize back to pixels.
    pixel_boxes = [(b[0] * w, b[1] * h, b[2] * w, b[3] * h) for b in fused_norm_boxes]
    return pixel_boxes, fused_scores, fused_labels
