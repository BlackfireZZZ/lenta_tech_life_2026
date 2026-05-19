"""Aggregator tests: per-field voting, fuzzy buckets, cross-track dedup."""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402

from price_tag_pipeline.aggregator import TrackAggregator, dedup_final_tags  # noqa: E402
from price_tag_pipeline.config import AggregationConfig  # noqa: E402
from price_tag_pipeline.types import (  # noqa: E402
    CropCandidate,
    FinalTag,
    ParsedTag,
    TagObservation,
    WeightUnit,
)


def _crop(image, bbox, *, area=15_000, conf=0.8) -> CropCandidate:
    return CropCandidate(
        image=image,
        sharpness=100.0,
        area_px=area,
        bbox_xyxy=bbox,
        detection_confidence=conf,
        frame_idx=0,
        timestamp_s=0.0,
    )


def _sharp_img(seed=0):
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, size=(120, 200, 3), dtype=np.uint8)


def _cfg(min_obs: int = 2, min_conf: float = 0.4) -> AggregationConfig:
    return AggregationConfig(
        min_observations_per_track=min_obs,
        min_final_confidence=min_conf,
        track_ttl_frames=30,
    )


def _obs(frame: int, track_id: int, regular: float | None, loyalty: float | None = None,
         conf: float = 0.9, sharpness: float = 100.0) -> TagObservation:
    parsed = ParsedTag(
        regular_price=regular,
        regular_price_confidence=conf if regular is not None else 0.0,
        loyalty_price=loyalty,
        loyalty_price_confidence=conf if loyalty is not None else 0.0,
        currency="RUB",
    )
    return TagObservation(
        frame_idx=frame,
        timestamp_s=frame / 30.0,
        track_id=track_id,
        bbox_xyxy=(10, 20, 100, 60),
        parsed=parsed,
        detection_confidence=0.85,
        sharpness=sharpness,
    )


def test_majority_wins_among_two_buckets():
    agg = TrackAggregator(_cfg())
    agg.add_observation(_obs(1, 7, regular=129.99))
    agg.add_observation(_obs(2, 7, regular=129.99))
    agg.add_observation(_obs(3, 7, regular=89.90))
    agg.mark_seen(7, 3, (10, 20, 100, 60))
    out = agg.flush_all()
    assert len(out) == 1
    assert out[0].regular_price == 129.99


def test_fuzzy_bucket_collapses_nearby_prices():
    # Two readings within tolerance should bucket together, not split.
    agg = TrackAggregator(AggregationConfig(
        min_observations_per_track=2, min_final_confidence=0.3,
        track_ttl_frames=30, price_fuzzy_tolerance=0.5,
    ))
    agg.add_observation(_obs(1, 5, regular=129.99))
    agg.add_observation(_obs(2, 5, regular=130.00))  # bucket with 129.99
    agg.mark_seen(5, 2, (0, 0, 10, 10))
    out = agg.flush_all()
    assert len(out) == 1
    # Center of the bucket is the rounded mean (within tolerance).
    assert abs(out[0].regular_price - 129.99) <= 0.5


def test_track_below_min_observations_dropped():
    agg = TrackAggregator(_cfg(min_obs=3))
    agg.add_observation(_obs(1, 5, regular=99.99))
    agg.add_observation(_obs(2, 5, regular=99.99))
    agg.mark_seen(5, 2, (0, 0, 10, 10))
    out = agg.flush_all()
    assert out == []


def test_dedup_merges_overlapping_same_price_tracks():
    a = FinalTag(track_id=1, bbox_xyxy=(0, 0, 100, 100), timestamp_s=1.0,
                 source_frames=[10, 11, 12], regular_price=129.99, overall_confidence=0.8, n_observations=3)
    b = FinalTag(track_id=2, bbox_xyxy=(2, 2, 102, 102), timestamp_s=1.1,
                 source_frames=[40, 41], regular_price=129.99, overall_confidence=0.7, n_observations=2)
    merged = dedup_final_tags([a, b], iou_threshold=0.4, time_window_s=2.0)
    assert len(merged) == 1
    # Source frames merged.
    assert set(merged[0].source_frames) == {10, 11, 12, 40, 41}


def test_dedup_keeps_distinct_when_prices_differ():
    a = FinalTag(track_id=1, bbox_xyxy=(0, 0, 100, 100), timestamp_s=1.0,
                 source_frames=[10], regular_price=99.99, overall_confidence=0.8, n_observations=1)
    b = FinalTag(track_id=2, bbox_xyxy=(0, 0, 100, 100), timestamp_s=1.0,
                 source_frames=[12], regular_price=149.99, overall_confidence=0.7, n_observations=1)
    merged = dedup_final_tags([a, b], iou_threshold=0.4, time_window_s=2.0)
    assert len(merged) == 2


# --- The moving-camera ID-switch scenario (the reason content-keyed dedup
#     exists): one physical tag fragments into two tracks whose boxes do NOT
#     overlap because the robot moved between them. The old IoU>thr gate left
#     the duplicate uncollapsed. ---

def test_dedup_same_barcode_merges_despite_zero_iou_and_time_gap():
    a = FinalTag(track_id=1, bbox_xyxy=(0, 0, 50, 50), timestamp_s=1.0,
                 source_frames=[10, 11], regular_price=129.99, overall_confidence=0.6,
                 n_observations=2, extra_fields={"barcode": "4601234567890"})
    b = FinalTag(track_id=2, bbox_xyxy=(900, 900, 960, 960), timestamp_s=31.0,
                 source_frames=[600, 601], regular_price=129.99, overall_confidence=0.9,
                 n_observations=2, extra_fields={"qr_code_barcode": "4601234567890"})
    merged = dedup_final_tags([a, b], iou_threshold=0.4, time_window_s=8.0)
    assert len(merged) == 1
    assert set(merged[0].source_frames) == {10, 11, 600, 601}
    # Representative must carry the barcode (GT primary key).
    bc = merged[0].extra_fields.get("barcode") or merged[0].extra_fields.get("qr_code_barcode")
    assert bc == "4601234567890"


def test_dedup_different_barcodes_never_merged_even_overlapping():
    a = FinalTag(track_id=1, bbox_xyxy=(0, 0, 100, 100), timestamp_s=1.0,
                 source_frames=[10], regular_price=99.99, overall_confidence=0.8,
                 n_observations=1, extra_fields={"barcode": "4601111111111"})
    b = FinalTag(track_id=2, bbox_xyxy=(0, 0, 100, 100), timestamp_s=1.0,
                 source_frames=[11], regular_price=99.99, overall_confidence=0.7,
                 n_observations=1, extra_fields={"barcode": "4602222222222"})
    merged = dedup_final_tags([a, b], iou_threshold=0.4, time_window_s=8.0)
    assert len(merged) == 2


def test_dedup_low_iou_same_price_and_name_merges():
    # No barcode (the common in-motion case). Camera moved → IoU 0, but price
    # AND name agree → same physical tag.
    a = FinalTag(track_id=1, bbox_xyxy=(0, 0, 100, 100), timestamp_s=1.0,
                 source_frames=[10], regular_price=129.99, product_name="Молоко Простоквашино 3.2%",
                 overall_confidence=0.7, n_observations=1)
    b = FinalTag(track_id=2, bbox_xyxy=(600, 0, 700, 100), timestamp_s=1.4,
                 source_frames=[18], regular_price=129.99, product_name="Молоко Простоквашино 3.2%",
                 overall_confidence=0.6, n_observations=1)
    merged = dedup_final_tags([a, b], iou_threshold=0.4, time_window_s=8.0)
    assert len(merged) == 1


def test_dedup_low_iou_same_price_diff_name_kept_distinct():
    # Precision guard: two different products that merely share a price must
    # NOT be merged when boxes don't overlap.
    a = FinalTag(track_id=1, bbox_xyxy=(0, 0, 100, 100), timestamp_s=1.0,
                 source_frames=[10], regular_price=129.99, product_name="Молоко Простоквашино",
                 overall_confidence=0.7, n_observations=1)
    b = FinalTag(track_id=2, bbox_xyxy=(600, 0, 700, 100), timestamp_s=1.4,
                 source_frames=[18], regular_price=129.99, product_name="Хлеб Бородинский",
                 overall_confidence=0.6, n_observations=1)
    merged = dedup_final_tags([a, b], iou_threshold=0.4, time_window_s=8.0)
    assert len(merged) == 2


def test_dedup_no_barcode_outside_time_window_not_merged():
    a = FinalTag(track_id=1, bbox_xyxy=(0, 0, 100, 100), timestamp_s=1.0,
                 source_frames=[10], regular_price=129.99, product_name="Молоко",
                 overall_confidence=0.7, n_observations=1)
    b = FinalTag(track_id=2, bbox_xyxy=(0, 0, 100, 100), timestamp_s=40.0,
                 source_frames=[800], regular_price=129.99, product_name="Молоко",
                 overall_confidence=0.6, n_observations=1)
    merged = dedup_final_tags([a, b], iou_threshold=0.4, time_window_s=8.0)
    assert len(merged) == 2


def test_finaltag_reports_best_frame_bbox_not_last_seen():
    # P0 #3: the emitted bbox/timestamp must be the strongest-recognition
    # frame, not state.last_bbox (the tag leaving the frame edge).
    agg = TrackAggregator(_cfg(min_obs=2, min_conf=0.0))
    best = TagObservation(
        frame_idx=5, timestamp_s=5 / 30.0, track_id=3,
        bbox_xyxy=(100, 100, 300, 260),
        parsed=ParsedTag(regular_price=99.99, regular_price_confidence=0.95, currency="RUB"),
        detection_confidence=0.95, sharpness=400.0,
    )
    weak = TagObservation(
        frame_idx=20, timestamp_s=20 / 30.0, track_id=3,
        bbox_xyxy=(1, 1, 40, 30),  # tiny, near origin: the leaving-frame box
        parsed=ParsedTag(regular_price=99.99, regular_price_confidence=0.30, currency="RUB"),
        detection_confidence=0.30, sharpness=15.0,
    )
    agg.add_observation(best)
    agg.add_observation(weak)
    # last_seen is the weak, leaving-frame box — must NOT be what we report.
    agg.mark_seen(3, 30, (0, 0, 20, 15))
    out = agg.flush_all()
    assert len(out) == 1
    assert out[0].bbox_xyxy == (100, 100, 300, 260)
    assert abs(out[0].timestamp_s - 5 / 30.0) < 1e-6


# --- P1 #6: per-track best-crop ranking (tracker-layer frame selection) ---

def test_best_crop_prefers_sharper_frame():
    import cv2
    agg = TrackAggregator(_cfg())
    sharp = _sharp_img(1)
    blurred = cv2.GaussianBlur(sharp, (9, 9), 0)
    bbox = (200, 200, 400, 360)  # interior, identical for both
    agg.push_crop(7, _crop(blurred, bbox), frame_w=1920, frame_h=1080)
    agg.push_crop(7, _crop(sharp, bbox), frame_w=1920, frame_h=1080)
    top = agg.best_crops(7, 1)
    assert len(top) == 1
    assert top[0].crop.image is sharp


def test_best_crop_penalizes_border_box():
    # Same image/area/conf; the box hugging the frame edge (tag leaving) must
    # rank below the interior one.
    agg = TrackAggregator(_cfg())
    img = _sharp_img(2)
    interior = _crop(img, (800, 400, 1000, 560))
    at_edge = _crop(img, (0, 400, 200, 560))  # x1 == 0 → truncated
    agg.push_crop(9, at_edge, frame_w=1920, frame_h=1080)
    agg.push_crop(9, interior, frame_w=1920, frame_h=1080)
    top = agg.best_crops(9, 2)
    assert top[0].crop is interior
    assert top[0].quality_score > top[1].quality_score


def test_border_factor_noop_when_frame_size_unknown():
    # Back-compat: no frame dims → no border penalty, still ranks by focus.
    agg = TrackAggregator(_cfg())
    img = _sharp_img(3)
    agg.push_crop(11, _crop(img, (0, 0, 50, 50)))  # would be penalised if known
    top = agg.best_crops(11, 1)
    assert top[0].quality_score > 0.0


def test_extra_fields_are_voted_into_final_tag():
    agg = TrackAggregator(_cfg())
    parsed = ParsedTag(
        extra_fields={"qr_code_barcode": "4601234567890", "price1_qr": 129.99},
        extra_confidences={"qr_code_barcode": 0.97, "price1_qr": 0.97},
    )
    for frame in (1, 2):
        agg.add_observation(TagObservation(
            frame_idx=frame,
            timestamp_s=frame / 30.0,
            track_id=9,
            bbox_xyxy=(10, 20, 100, 60),
            parsed=parsed,
            detection_confidence=0.9,
            sharpness=120.0,
        ))
    agg.mark_seen(9, 2, (10, 20, 100, 60))
    out = agg.flush_all()
    assert len(out) == 1
    row = out[0].to_dict()
    assert row["qr_code_barcode"] == "4601234567890"
    assert row["price1_qr"] == 129.99


def test_discount_amount_percentage_string_survives_voting():
    """Regression: discount_amount valid values carry a unit symbol

    ("-18%", "-286₽" — price-tag-guide §4). It must NOT be numeric-voted, or
    _coerce_float drops every read and the scored field is emitted empty.
    """
    from price_tag_pipeline.submission import final_tags_to_csv

    agg = TrackAggregator(_cfg())
    parsed = ParsedTag(
        extra_fields={"discount_amount": "-18%"},
        extra_confidences={"discount_amount": 0.9},
    )
    for frame in (1, 2):
        agg.add_observation(TagObservation(
            frame_idx=frame,
            timestamp_s=frame / 30.0,
            track_id=3,
            bbox_xyxy=(10, 20, 100, 60),
            parsed=parsed,
            detection_confidence=0.9,
            sharpness=120.0,
        ))
    agg.mark_seen(3, 2, (10, 20, 100, 60))
    out = agg.flush_all()
    assert len(out) == 1
    assert out[0].to_dict()["discount_amount"] == "-18%"
    # And it must reach the graded CSV verbatim (not "" and not "-18.00").
    csv_text = final_tags_to_csv(out, filename="25_2-10")
    header, row = csv_text.splitlines()[0], csv_text.splitlines()[1]
    col = header.split(",").index("discount_amount")
    assert row.split(",")[col] == "-18%"


def test_ruble_discount_amount_string_survives_voting():
    agg = TrackAggregator(_cfg())
    parsed = ParsedTag(
        extra_fields={"discount_amount": "-286₽"},
        extra_confidences={"discount_amount": 0.9},
    )
    for frame in (1, 2):
        agg.add_observation(TagObservation(
            frame_idx=frame, timestamp_s=frame / 30.0, track_id=4,
            bbox_xyxy=(10, 20, 100, 60), parsed=parsed,
            detection_confidence=0.9, sharpness=120.0,
        ))
    agg.mark_seen(4, 2, (10, 20, 100, 60))
    out = agg.flush_all()
    assert len(out) == 1
    assert out[0].to_dict()["discount_amount"] == "-286₽"
