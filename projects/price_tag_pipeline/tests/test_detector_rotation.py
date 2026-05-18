"""Box un-projection for the 90° upright-rotation detector path.

The Lenta scan-robot cam is mounted 90° CW; the detector runs on an
upright-rotated frame and boxes are mapped back to original-frame coords by
``unrotate_box_xyxy``. These tests pin that inverse against the *forward*
cv2/np.rot90 pixel map so a sign error can't ship to main. Pure — no model,
no OpenCV needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from price_tag_pipeline.detector import unrotate_box_xyxy  # noqa: E402


def _forward(box, rotation, ow, oh):
    """Where (x1,y1,x2,y2) lands after rotating the frame for inference.

    Continuous (box-edge) convention, image *size* not size-1:
    CCW ``x_r=y, y_r=ow-x``; CW ``x_r=oh-y, y_r=x``.
    """
    x1, y1, x2, y2 = box
    if rotation == "ccw":
        pts = [(y1, ow - x1), (y2, ow - x2)]
    else:  # cw
        pts = [(oh - y1, x1), (oh - y2, x2)]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def test_ccw_roundtrip_non_square():
    ow, oh = 4000, 2000
    box = (100, 200, 300, 500)
    rotated = _forward(box, "ccw", ow, oh)
    assert unrotate_box_xyxy(rotated, "ccw", ow, oh) == box


def test_cw_roundtrip_non_square():
    ow, oh = 4000, 2000
    box = (100, 200, 300, 500)
    rotated = _forward(box, "cw", ow, oh)
    assert unrotate_box_xyxy(rotated, "cw", ow, oh) == box


def test_full_frame_box_maps_to_full_frame():
    ow, oh = 3840, 2160
    # rotated frame is oh x ow; a full-cover box must come back full-cover
    assert unrotate_box_xyxy((0, 0, oh, ow), "ccw", ow, oh) == (0, 0, ow, oh)
    assert unrotate_box_xyxy((0, 0, oh, ow), "cw", ow, oh) == (0, 0, ow, oh)


def test_none_is_identity_and_clamped():
    assert unrotate_box_xyxy((10, 20, 30, 40), "none", 100, 100) == (10, 20, 30, 40)
    # out-of-frame coords are clamped, order normalized
    assert unrotate_box_xyxy((-5, 50, 30, 10), "none", 100, 100) == (0, 10, 30, 50)


def test_corners_distinct_under_each_rotation():
    ow, oh = 1920, 1080
    box = (0, 0, 200, 100)  # top-left tag
    ccw = unrotate_box_xyxy(_forward(box, "ccw", ow, oh), "ccw", ow, oh)
    cw = unrotate_box_xyxy(_forward(box, "cw", ow, oh), "cw", ow, oh)
    assert ccw == box and cw == box
