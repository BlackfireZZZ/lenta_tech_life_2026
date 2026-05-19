"""Product-card grouping tests (the P2 over-alerting fix).

rotation="none" so original==upright and the geometry intent is readable.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from price_tag_pipeline.shelf_analytics import (  # noqa: E402
    AssociationResult,
    CardConfig,
    TrackTrace,
    build_card_set,
    regroup_missing_price_tag,
)
from price_tag_pipeline.shelf_analytics.schema import TagProductRelation  # noqa: E402

NONE = CardConfig(rotation="none")


def _trace(tid: int, box, f0: int, f1: int) -> TrackTrace:
    return TrackTrace(track_id=tid, boxes={f: tuple(box) for f in range(f0, f1)})


def test_adjacent_identical_facings_collapse_to_one_card() -> None:
    # Three same-size bottles in a row, same band, small gaps, co-visible.
    traces = [
        _trace(1, (100, 100, 150, 300), 0, 20),
        _trace(2, (160, 100, 210, 300), 0, 20),
        _trace(3, (220, 100, 270, 300), 0, 20),
    ]
    cs = build_card_set(traces, 640, 480, 25.0, "v", cfg=NONE)

    assert len(cs.cards) == 1
    assert cs.cards[0].facing_count == 3
    assert set(cs.card_members[cs.cards[0].card_id]) == {1, 2, 3}
    assert cs.track_to_card == {1: "card_v_0001", 2: "card_v_0001", 3: "card_v_0001"}


def test_far_apart_facings_stay_separate() -> None:
    traces = [
        _trace(1, (100, 100, 150, 300), 0, 20),
        _trace(2, (160, 100, 210, 300), 0, 20),
        _trace(9, (1200, 100, 1250, 300), 0, 20),  # big x gap
    ]
    cs = build_card_set(traces, 640, 480, 25.0, "v", cfg=NONE)
    assert len(cs.cards) == 2
    sizes = sorted(c.facing_count for c in cs.cards)
    assert sizes == [1, 2]


def test_different_shelf_band_not_merged() -> None:
    traces = [
        _trace(1, (100, 100, 150, 300), 0, 20),
        _trace(2, (160, 600, 210, 800), 0, 20),  # far lower band
    ]
    cs = build_card_set(traces, 640, 480, 25.0, "v", cfg=NONE)
    assert len(cs.cards) == 2


def test_appearance_gate_splits_dissimilar_neighbours() -> None:
    traces = [
        _trace(1, (100, 100, 150, 300), 0, 20),
        _trace(2, (160, 100, 210, 300), 0, 20),
    ]
    appearance = {1: (1.0, 0.0, 0.0), 2: (0.0, 1.0, 0.0)}  # orthogonal
    cs = build_card_set(traces, 640, 480, 25.0, "v", appearance=appearance, cfg=NONE)
    assert len(cs.cards) == 2

    appearance_same = {1: (1.0, 0.0, 0.0), 2: (0.99, 0.01, 0.0)}
    cs2 = build_card_set(traces, 640, 480, 25.0, "v", appearance=appearance_same, cfg=NONE)
    assert len(cs2.cards) == 1


def test_large_frame_gap_not_merged() -> None:
    a = _trace(1, (100, 100, 150, 300), 0, 10)
    b = _trace(2, (160, 100, 210, 300), 200, 210)  # seen much later
    cs = build_card_set([a, b], 640, 480, 25.0, "v", cfg=NONE)
    assert len(cs.cards) == 2


def test_best_track_picks_representative() -> None:
    traces = [
        _trace(1, (100, 100, 150, 300), 0, 5),
        _trace(2, (160, 100, 210, 300), 0, 20),
    ]
    cs = build_card_set(traces, 640, 480, 25.0, "v",
                        best_track=lambda tid: 99.0 if tid == 1 else 1.0, cfg=NONE)
    assert len(cs.cards) == 1
    assert cs.cards[0].product_track_id == 1  # chosen by best_track, not length


def test_regroup_collapses_facing_level_missing_tag() -> None:
    # Group A = {1,2,3}, Group B = {9}. One ok relation covers track 2 -> A.
    traces = [
        _trace(1, (100, 100, 150, 300), 0, 20),
        _trace(2, (160, 100, 210, 300), 0, 20),
        _trace(3, (220, 100, 270, 300), 0, 20),
        _trace(9, (1200, 100, 1250, 300), 0, 20),
    ]
    cs = build_card_set(traces, 640, 480, 25.0, "v", cfg=NONE)
    assoc = AssociationResult(
        relations=(TagProductRelation(price_tag_id="tag_7",
                                      product_group_id="product_2",
                                      score=0.8, status="ok"),),
        unmatched_price_tags=(),
        unmatched_products=(),
    )
    missing = regroup_missing_price_tag(assoc, cs)
    # Facing-level would flag tracks 1,3,9; group-level flags only group B.
    assert len(missing) == 1
    assert cs.card_members[missing[0]] == (9,)
