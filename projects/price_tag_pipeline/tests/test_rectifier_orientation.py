"""Crop un-rotation sign for the 90°-CW robot cam.

The detector runs on an upright-rotated frame but un-projects boxes to the
ORIGINAL (sideways) frame; the pipeline crops from that original frame, so the
rectifier must un-rotate every crop upright by the camera-mounting inverse —
the SAME direction the detector / make_shelf_audit_fixture use
(``ccw`` profile, cam 90° CW → ``ROTATE_90_COUNTERCLOCKWISE``). A CW rotation
there ships Qwen-VL upside-down crops. These pin the sign so it can't regress.
Pure — needs only numpy + cv2.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from price_tag_pipeline.rectifier import _upright_crop  # noqa: E402


def _asym() -> np.ndarray:
    """A wide, orientation-distinguishable image (every 90° turn differs)."""
    img = np.zeros((30, 90, 3), dtype=np.uint8)
    img[0, :, 0] = 255  # top row red
    img[:, 0, 1] = 255  # left col green
    return img


def test_ccw_profile_un_rotates_counter_clockwise():
    img = _asym()
    out = _upright_crop(img, "ccw", rotate_vertical_tags=True)
    assert np.array_equal(out, cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))
    # Explicitly NOT the old clockwise behaviour (that was 180° / upside-down).
    assert not np.array_equal(out, cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))


def test_cw_profile_un_rotates_clockwise():
    img = _asym()
    out = _upright_crop(img, "cw", rotate_vertical_tags=True)
    assert np.array_equal(out, cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))


def test_rotated_profile_master_switch_off_is_noop():
    img = _asym()
    assert np.array_equal(_upright_crop(img, "ccw", rotate_vertical_tags=False), img)


def test_none_profile_keeps_legacy_tall_to_cw_heuristic():
    tall = np.zeros((90, 30, 3), dtype=np.uint8)  # h > w*1.6
    tall[0, :, 0] = 255
    out = _upright_crop(tall, "none", rotate_vertical_tags=True)
    assert np.array_equal(out, cv2.rotate(tall, cv2.ROTATE_90_CLOCKWISE))


def test_none_profile_leaves_wide_crop_untouched():
    wide = _asym()  # already upright/landscape
    assert np.array_equal(_upright_crop(wide, "none", rotate_vertical_tags=True), wide)
