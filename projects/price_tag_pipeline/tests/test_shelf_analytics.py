"""Shelf analytics layer tests.

These lock the optional product/facing path while keeping it independent from
the existing price-tag CSV pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from price_tag_pipeline.shelf_analytics import (  # noqa: E402
    MatchingConfig,
    PriceTagObservation,
    ProductFacing,
    build_shelf_state,
    build_shelf_state_from_pipeline_outputs,
    group_facings,
    match_price_tags,
    product_facing_from_detection,
)
from price_tag_pipeline.shelf_analytics.pipeline import price_tag_from_final_tag  # noqa: E402
from price_tag_pipeline.types import Detection, FinalTag  # noqa: E402


def test_group_facings_uses_known_sku_without_making_detector_classes() -> None:
    facings = [
        ProductFacing(id="p1", bbox_xyxy=(10, 10, 50, 110), row_id="r1", sku_id="sku_a", sku_confidence=0.91),
        ProductFacing(id="p2", bbox_xyxy=(54, 10, 94, 110), row_id="r1", sku_id="sku_a", sku_confidence=0.88),
        ProductFacing(id="p3", bbox_xyxy=(180, 10, 220, 110), row_id="r1", sku_id="sku_b", sku_confidence=0.93),
    ]

    groups = group_facings(facings)

    assert [g.facing_count for g in groups] == [2, 1]
    assert groups[0].sku_id == "sku_a"
    assert groups[0].member_facing_ids == ("p1", "p2")
    assert groups[1].sku_id == "sku_b"


def test_group_facings_can_cluster_unknown_products_by_embedding() -> None:
    facings = [
        ProductFacing(id="p1", bbox_xyxy=(10, 10, 50, 110), row_id="r1", embedding=(1.0, 0.0, 0.0)),
        ProductFacing(id="p2", bbox_xyxy=(55, 10, 95, 110), row_id="r1", embedding=(0.99, 0.01, 0.0)),
        ProductFacing(id="p3", bbox_xyxy=(150, 10, 190, 110), row_id="r1", embedding=(0.0, 1.0, 0.0)),
    ]

    groups = group_facings(facings)

    assert [g.facing_count for g in groups] == [2, 1]
    assert groups[0].status == "unknown_sku"


def test_match_price_tags_prefers_same_row_below_overlap() -> None:
    state = build_shelf_state(
        image_id="frame_001",
        product_facings=[
            ProductFacing(
                id="p1",
                bbox_xyxy=(100, 100, 160, 260),
                row_id="r1",
                sku_id="milk_1l",
                sku_confidence=0.9,
                display_name="Молоко 1л",
            ),
            ProductFacing(
                id="p2",
                bbox_xyxy=(165, 100, 225, 260),
                row_id="r1",
                sku_id="milk_1l",
                sku_confidence=0.9,
                display_name="Молоко 1л",
            ),
        ],
        price_tags=[
            PriceTagObservation(
                id="t1",
                bbox_xyxy=(95, 270, 230, 305),
                row_id="r1",
                price=99.9,
                product_name="Молоко пастеризованное 1л",
            )
        ],
    )

    assert len(state.groups) == 1
    assert state.relations[0].status == "ok"
    assert state.relations[0].product_group_id == state.groups[0].id
    assert state.to_dict()["relations"][0]["score"] > 0.42


def test_match_price_tags_marks_close_candidates_ambiguous() -> None:
    groups = group_facings(
        [
            ProductFacing(id="p1", bbox_xyxy=(100, 100, 160, 260), row_id="r1"),
            ProductFacing(id="p2", bbox_xyxy=(170, 100, 230, 260), row_id="r1"),
        ]
    )
    tag = PriceTagObservation(id="t1", bbox_xyxy=(125, 270, 205, 305), row_id="r1")

    relations = match_price_tags(groups, [tag], MatchingConfig(min_score=0.2, ambiguous_margin=0.2))

    assert relations[0].status == "ambiguous"
    assert relations[0].product_group_id is None


def test_price_tag_adapter_keeps_existing_final_tag_contract() -> None:
    final = FinalTag(
        track_id=7,
        bbox_xyxy=(1, 2, 30, 40),
        timestamp_s=12.5,
        source_frames=[10, 11],
        regular_price=129.99,
        loyalty_price=99.99,
        product_name="Сыр",
        extra_fields={"barcode": "460123"},
        overall_confidence=0.87,
    )

    tag = price_tag_from_final_tag(final)

    assert tag.id == "tag_7"
    assert tag.price == 129.99
    assert tag.barcode == "460123"
    assert tag.source_track_id == 7


def test_existing_detection_and_final_tag_outputs_can_feed_shelf_state() -> None:
    det = Detection(
        frame_idx=3,
        timestamp_s=0.12,
        bbox_xyxy=(100, 100, 160, 260),
        confidence=0.8,
        class_id=0,
        class_name="product_facing",
        track_id=42,
    )
    final = FinalTag(
        track_id=9,
        bbox_xyxy=(95, 270, 170, 305),
        timestamp_s=0.14,
        source_frames=[3],
        regular_price=79.9,
        product_name="Молоко",
        overall_confidence=0.76,
    )

    facing = product_facing_from_detection(det, row_id="r1")
    state = build_shelf_state_from_pipeline_outputs("frame_003", [det], [final])

    assert facing.id == "product_42_0"
    assert state.product_facings[0].id == "product_42_0"
    assert state.price_tags[0].id == "tag_9"
    assert state.groups[0].facing_count == 1
