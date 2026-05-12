"""WBF unit tests — make sure the fallback path is correct."""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from price_tag_pipeline.inference.wbf import weighted_box_fusion  # noqa: E402


def test_two_overlapping_boxes_get_fused():
    a = [(0.10, 0.10, 0.30, 0.30)]
    b = [(0.12, 0.12, 0.32, 0.32)]
    boxes, scores, labels = weighted_box_fusion(
        [a, b], [[0.9], [0.8]], [[0], [0]],
        iou_threshold=0.5, skip_box_threshold=0.01,
    )
    assert len(boxes) == 1
    # Fused box is between the two inputs.
    x1, y1, x2, y2 = boxes[0]
    assert 0.10 < x1 < 0.13
    assert 0.30 < x2 < 0.33


def test_distant_boxes_kept_separate():
    a = [(0.05, 0.05, 0.10, 0.10)]
    b = [(0.80, 0.80, 0.90, 0.90)]
    boxes, scores, labels = weighted_box_fusion(
        [a, b], [[0.9], [0.9]], [[0], [0]],
        iou_threshold=0.5,
    )
    assert len(boxes) == 2


def test_different_labels_kept_separate():
    a = [(0.10, 0.10, 0.30, 0.30)]
    b = [(0.10, 0.10, 0.30, 0.30)]
    boxes, scores, labels = weighted_box_fusion(
        [a, b], [[0.9], [0.9]], [[0], [1]],
        iou_threshold=0.5,
    )
    assert len(boxes) == 2
    assert set(labels) == {0, 1}


def test_skip_threshold_drops_low_scores():
    a = [(0.10, 0.10, 0.30, 0.30)]
    b = [(0.10, 0.10, 0.30, 0.30)]
    boxes, scores, labels = weighted_box_fusion(
        [a, b], [[0.9], [0.005]], [[0], [0]],
        iou_threshold=0.5, skip_box_threshold=0.01,
    )
    assert len(boxes) == 1
