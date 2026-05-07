from price_tag_pipeline.aggregator import TrackAggregator
from price_tag_pipeline.config import AggregationConfig
from price_tag_pipeline.types import ParsedPrice, PriceObservation


def _obs(frame: int, price: str, conf: float) -> PriceObservation:
    return PriceObservation(
        frame_idx=frame,
        timestamp_s=frame / 30.0,
        track_id=7,
        bbox_xyxy=(10, 20, 100, 60),
        parsed=ParsedPrice(
            raw=price,
            normalized_price=price,
            value=float(price),
            currency="EUR",
            unit=None,
            confidence=conf,
        ),
        detection_confidence=0.9,
        ocr_confidence=0.85,
        sharpness=120.0,
    )


def test_weighted_vote_returns_stable_price():
    agg = TrackAggregator(
        AggregationConfig(min_observations_per_track=3, min_final_confidence=0.5, track_ttl_frames=10)
    )
    agg.add_observation(_obs(1, "2.49", 0.90))
    agg.add_observation(_obs(2, "2.49", 0.88))
    agg.add_observation(_obs(3, "2.99", 0.45))
    out = agg.flush_all()
    assert len(out) == 1
    assert out[0].price == "2.49"
    assert out[0].track_id == 7

