"""Track-level shelf-audit associator tests.

Two concerns, tested separately for clarity:
  * geometry.to_upright is the exact inverse of detector.unrotate_box_xyxy
    (the P0-critical coordinate fact);
  * associate_tracks logic (relations / OOS / missing-tag / ambiguous /
    persistence / min-co-frames), exercised in identity space so the geometry
    intent is readable.

This path is independent of the graded 29-column CSV.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from price_tag_pipeline.detector import unrotate_box_xyxy  # noqa: E402
from price_tag_pipeline.shelf_analytics import (  # noqa: E402
    AssociationConfig,
    TrackTrace,
    associate_tracks,
)
from price_tag_pipeline.shelf_analytics.geometry import to_upright  # noqa: E402


# --------------------------------------------------------------------------
# Coordinate fact: to_upright is the inverse of unrotate_box_xyxy
# --------------------------------------------------------------------------

def test_to_upright_inverts_unrotate_ccw() -> None:
    orig_w, orig_h = 1920, 1080  # original (sideways) frame; rotated is H x W
    rotated_box = (120.0, 300.0, 480.0, 700.0)
    orig = unrotate_box_xyxy(rotated_box, "ccw", orig_w, orig_h)
    back = to_upright(orig, orig_w, orig_h, "ccw")
    assert all(abs(a - b) <= 1.0 for a, b in zip(back, rotated_box))


def test_to_upright_inverts_unrotate_cw() -> None:
    orig_w, orig_h = 1920, 1080
    rotated_box = (50.0, 60.0, 240.0, 880.0)
    orig = unrotate_box_xyxy(rotated_box, "cw", orig_w, orig_h)
    back = to_upright(orig, orig_w, orig_h, "cw")
    assert all(abs(a - b) <= 1.0 for a, b in zip(back, rotated_box))


def test_to_upright_none_is_identity() -> None:
    assert to_upright((10.0, 20.0, 30.0, 40.0), 100, 100, "none") == (10.0, 20.0, 30.0, 40.0)


# --------------------------------------------------------------------------
# Associator logic (rotation="none" -> boxes are already upright)
# --------------------------------------------------------------------------

NONE = AssociationConfig(rotation="none")


def _trace(tid: int, box, f0: int, f1: int) -> TrackTrace:
    return TrackTrace(track_id=tid, boxes={f: tuple(box) for f in range(f0, f1)})


def test_clean_tag_below_product_is_ok() -> None:
    prod = _trace(1, (100, 100, 160, 260), 0, 15)
    tag = _trace(7, (95, 270, 230, 305), 0, 15)

    res = associate_tracks([tag], [prod], frame_w=640, frame_h=480, fps=25.0, cfg=NONE)

    assert len(res.relations) == 1
    rel = res.relations[0]
    assert rel.status == "ok"
    assert rel.price_tag_id == "tag_7"
    assert rel.product_group_id == "product_1"
    assert "tag_below_product" in rel.reasons
    assert res.unmatched_price_tags == ()
    assert res.unmatched_products == ()


def test_price_tag_with_no_product_is_out_of_stock_candidate() -> None:
    tag = _trace(7, (95, 270, 230, 305), 0, 15)  # 15 frames -> persistent

    res = associate_tracks([tag], [], frame_w=640, frame_h=480, fps=25.0, cfg=NONE)

    assert res.relations[0].status == "ambiguous"
    assert res.relations[0].product_group_id is None
    assert [u.track_id for u in res.unmatched_price_tags] == [7]
    assert res.unmatched_price_tags[0].persistent is True
    assert res.unmatched_products == ()


def test_product_with_no_tag_is_missing_price_tag_candidate() -> None:
    prod = _trace(3, (100, 100, 160, 260), 0, 15)

    res = associate_tracks([], [prod], frame_w=640, frame_h=480, fps=25.0, cfg=NONE)

    assert res.relations == ()
    assert [u.track_id for u in res.unmatched_products] == [3]
    assert res.unmatched_products[0].persistent is True
    assert res.unmatched_products[0].kind == "product"


def test_two_equally_good_products_are_ambiguous() -> None:
    # Symmetric around the tag's x-centre -> near-equal scores -> ambiguous.
    tag = _trace(7, (100, 270, 200, 305), 0, 15)
    left = _trace(1, (60, 100, 150, 260), 0, 15)
    right = _trace(2, (150, 100, 240, 260), 0, 15)

    res = associate_tracks([tag], [left, right], frame_w=640, frame_h=480,
                            fps=25.0, cfg=AssociationConfig(rotation="none",
                                                            ambiguous_margin=0.15))

    assert res.relations[0].status == "ambiguous"
    assert res.relations[0].product_group_id is None
    assert [u.track_id for u in res.unmatched_price_tags] == [7]


def test_persistence_gate_marks_short_tracks_non_persistent() -> None:
    short = _trace(7, (95, 270, 230, 305), 0, 4)   # 4 frames < default 10
    res = associate_tracks([short], [], frame_w=640, frame_h=480, fps=25.0, cfg=NONE)
    assert res.unmatched_price_tags[0].persistent is False
    assert res.unmatched_price_tags[0].n_frames == 4


def test_min_co_frames_blocks_thin_overlap() -> None:
    # Product 0..1 and tag 0..1: only 2 shared frames < min_co_frames(3).
    prod = _trace(1, (100, 100, 160, 260), 0, 2)
    tag = _trace(7, (95, 270, 230, 305), 0, 2)

    res = associate_tracks([tag], [prod], frame_w=640, frame_h=480, fps=25.0, cfg=NONE)

    assert res.relations[0].status == "ambiguous"  # never scored
    assert [u.track_id for u in res.unmatched_price_tags] == [7]
    assert [u.track_id for u in res.unmatched_products] == [1]


def test_unmatched_track_timestamps_use_fps() -> None:
    tag = _trace(7, (95, 270, 230, 305), 50, 65)  # frames 50..64 @ 25 fps
    res = associate_tracks([tag], [], frame_w=640, frame_h=480, fps=25.0, cfg=NONE)
    u = res.unmatched_price_tags[0]
    assert u.first_frame == 50 and u.last_frame == 64
    assert abs(u.first_s - 2.0) < 1e-6
    assert abs(u.last_s - 64 / 25.0) < 1e-6
    d = u.to_dict()
    assert d["kind"] == "price_tag" and d["persistent"] is True
