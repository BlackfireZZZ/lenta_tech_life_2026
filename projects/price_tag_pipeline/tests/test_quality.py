"""Focus-measure tests.

``tenengrad_sharpness`` only needs to be monotone in real focus (sharper >
blurrier > flat) — the tracker uses it for *relative* crop ranking, so the
absolute scale is irrelevant.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2  # noqa: E402

from price_tag_pipeline.quality import tenengrad_sharpness  # noqa: E402


def _noise(seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, size=(120, 200, 3), dtype=np.uint8)


def test_empty_image_is_zero():
    assert tenengrad_sharpness(np.zeros((0, 0, 3), dtype=np.uint8)) == 0.0


def test_flat_image_has_near_zero_focus():
    flat = np.full((120, 200, 3), 127, dtype=np.uint8)
    assert tenengrad_sharpness(flat) < 1.0


def test_sharper_beats_blurred_beats_flat():
    sharp = _noise()
    blurred = cv2.GaussianBlur(sharp, (9, 9), 0)
    flat = np.full_like(sharp, 127)
    s = tenengrad_sharpness(sharp)
    b = tenengrad_sharpness(blurred)
    f = tenengrad_sharpness(flat)
    assert s > b > f


def test_grayscale_input_supported():
    gray = _noise()[:, :, 0].copy()
    assert tenengrad_sharpness(gray) > 0.0
