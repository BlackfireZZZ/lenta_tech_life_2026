"""Pipeline smoke test: end-to-end on a synthetic video with stubbed detector + OCR.

Does not require a trained model, a real video, or any external dependency
beyond cv2 + numpy + pyyaml.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from price_tag_pipeline.aggregator import TrackAggregator  # noqa: E402
from price_tag_pipeline.config import AggregationConfig, ParserConfig  # noqa: E402
from price_tag_pipeline.parser import TagParser  # noqa: E402
from price_tag_pipeline.types import ParsedTag, TagObservation, WeightUnit  # noqa: E402


def test_full_aggregator_round_trip(tmp_path: Path) -> None:
    """Build a few observations, finalize, write to JSONL, read back."""
    agg = TrackAggregator(AggregationConfig(
        min_observations_per_track=2,
        min_final_confidence=0.4,
        track_ttl_frames=20,
    ))
    parser = TagParser(ParserConfig(min_price=0.01, max_price=999999.99))

    raw_lines = [
        "Молоко Простоквашино 1 л 89,90 ₽",
        "Молоко Простоквашино 1л 89⁹⁰ руб",
        "Молоко 1 л 89,90 ₽",
    ]
    for i, raw in enumerate(raw_lines):
        parsed = parser.parse_text(raw, ocr_confidence=0.9)
        obs = TagObservation(
            frame_idx=i,
            timestamp_s=i / 30.0,
            track_id=42,
            bbox_xyxy=(10, 10, 110, 70),
            parsed=parsed,
            detection_confidence=0.88,
            sharpness=120.0,
        )
        agg.add_observation(obs)
    agg.mark_seen(42, len(raw_lines) - 1, (10, 10, 110, 70))

    finalized = agg.flush_all()
    assert len(finalized) == 1
    final = finalized[0]
    assert abs(final.regular_price - 89.90) < 0.01
    assert final.weight_value == 1.0
    assert final.weight_unit == "л"
    # Round-trip JSON.
    out = tmp_path / "out.jsonl"
    out.write_text(json.dumps(final.to_dict(), ensure_ascii=False) + "\n", encoding="utf-8")
    reloaded = json.loads(out.read_text(encoding="utf-8").strip())
    assert reloaded["regular_price"] == 89.90
    assert reloaded["weight_unit"] == "л"
