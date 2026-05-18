"""Frame-sampling de-dup core — pure NumPy, no OpenCV / video needed."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from price_tag_pipeline.data.frame_sampling import (  # noqa: E402
    frame_signature,
    is_near_duplicate,
    signature_distance,
)


def _frame(value: int, h: int = 120, w: int = 200) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def test_signature_is_resolution_invariant_and_normalized():
    a = frame_signature(_frame(128, 120, 200))
    b = frame_signature(_frame(128, 480, 640))  # same content, bigger
    assert a.shape == b.shape  # fixed grid
    assert 0.0 <= float(a.min()) and float(a.max()) <= 1.0
    assert signature_distance(a, b) < 1e-6


def test_identical_frames_are_near_duplicates():
    a = frame_signature(_frame(100))
    b = frame_signature(_frame(100))
    assert signature_distance(a, b) == 0.0
    assert is_near_duplicate(a, b, min_diff=0.02) is True


def test_a_real_change_is_not_a_duplicate():
    parked = frame_signature(_frame(100))
    moved = frame_signature(_frame(180))  # whole scene shifted = big diff
    assert signature_distance(parked, moved) > 0.02
    assert is_near_duplicate(parked, moved, min_diff=0.02) is False


def test_min_diff_zero_disables_dedup():
    a = frame_signature(_frame(100))
    b = frame_signature(_frame(100))
    assert is_near_duplicate(a, b, min_diff=0.0) is False


def test_tiny_local_change_below_threshold_is_duplicate():
    base = _frame(100)
    nudged = base.copy()
    nudged[0:4, 0:4] = 255  # a few stray pixels — robot still parked
    a = frame_signature(base)
    b = frame_signature(nudged)
    assert 0.0 < signature_distance(a, b) < 0.02
    assert is_near_duplicate(a, b, min_diff=0.02) is True


def test_signature_distance_mismatched_shapes_is_max():
    assert signature_distance(np.zeros((4, 4)), np.zeros((5, 5))) == 1.0
