"""Aggregator tests: per-field voting, fuzzy buckets, cross-track dedup."""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from price_tag_pipeline.aggregator import TrackAggregator, dedup_final_tags  # noqa: E402
from price_tag_pipeline.config import AggregationConfig  # noqa: E402
from price_tag_pipeline.types import FinalTag, ParsedTag, TagObservation, WeightUnit  # noqa: E402


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
    merged = dedup_final_tags([a, b], iou_threshold=0.4, time_window_frames=50)
    assert len(merged) == 1
    # Source frames merged.
    assert set(merged[0].source_frames) == {10, 11, 12, 40, 41}


def test_dedup_keeps_distinct_when_prices_differ():
    a = FinalTag(track_id=1, bbox_xyxy=(0, 0, 100, 100), timestamp_s=1.0,
                 source_frames=[10], regular_price=99.99, overall_confidence=0.8, n_observations=1)
    b = FinalTag(track_id=2, bbox_xyxy=(0, 0, 100, 100), timestamp_s=1.0,
                 source_frames=[12], regular_price=149.99, overall_confidence=0.7, n_observations=1)
    merged = dedup_final_tags([a, b], iou_threshold=0.4, time_window_frames=50)
    assert len(merged) == 2
