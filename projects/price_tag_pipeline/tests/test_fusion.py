"""Multi-frame crop fusion (Level-2) — shape/robustness contract.

Decode quality is benchmarked separately (scripts/eval_code_reading.py
--fuse); here we only pin the pure contract: <2 crops → None; ≥2 →
a single uint8 image the reader can localise+decode on; mixed sizes are
tolerated; an uncorrelated frame doesn't crash fusion.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from price_tag_pipeline.fusion import fuse_crops  # noqa: E402


def _tag(seed: int, h: int = 120, w: int = 200) -> np.ndarray:
    rng = np.random.RandomState(seed)
    img = rng.randint(0, 60, size=(h, w, 3), dtype=np.uint8)
    img[30:90, 40:160] = 220  # a high-contrast "symbol" block, constant
    return img


def test_fewer_than_two_returns_none():
    assert fuse_crops([]) is None
    assert fuse_crops([_tag(0)]) is None


def test_two_plus_returns_single_uint8_image():
    base = _tag(1)
    crops = [base] + [
        (base + np.random.RandomState(i).randint(0, 8, base.shape, dtype=np.uint8))
        for i in range(4)
    ]
    out = fuse_crops(crops, max_frames=4)
    assert out is not None
    assert out.dtype == np.uint8 and out.ndim == 3


def test_mixed_sizes_tolerated():
    import cv2

    a = _tag(2)
    b = cv2.resize(a, (160, 96))
    out = fuse_crops([a, b, a.copy()], max_frames=3)
    assert out is not None
    # Reference is the sharpest crop; output matches a reference size.
    assert out.shape[2] == 3


def test_uncorrelated_frame_does_not_crash():
    a = _tag(3)
    junk = np.random.RandomState(9).randint(0, 256, a.shape, dtype=np.uint8)
    out = fuse_crops([a, a.copy(), junk], max_frames=3)
    # Either fuses the good pair or returns None — never raises.
    assert out is None or out.dtype == np.uint8
