"""Submission builder + schema validation tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from price_tag_pipeline.submission import (  # noqa: E402
    build_submission, _validate_tag, collect_predictions,
)


def test_validate_tag_ok():
    good = {
        "video_id": "vid_a", "track_id": 1, "bbox": [10, 20, 100, 60],
        "currency": "RUB", "regular_price": 99.99, "promo_flag": False,
    }
    assert _validate_tag(good) is None


def test_validate_tag_missing_required():
    bad = {"track_id": 1, "bbox": [10, 20, 100, 60], "currency": "RUB"}
    assert _validate_tag(bad) is not None  # video_id missing


def test_validate_tag_wrong_bbox_shape():
    bad = {"video_id": "v", "track_id": 1, "bbox": [10, 20, 100], "currency": "RUB"}
    assert _validate_tag(bad) is not None


def test_build_submission_from_two_videos(tmp_path: Path):
    inputs = tmp_path / "in"
    inputs.mkdir()
    (inputs / "vid_a.jsonl").write_text(
        json.dumps({"track_id": 1, "bbox": [10, 20, 100, 60],
                    "currency": "RUB", "regular_price": 99.99, "promo_flag": False}) + "\n",
        encoding="utf-8",
    )
    (inputs / "vid_b.jsonl").write_text(
        json.dumps({"track_id": 7, "bbox": [50, 60, 150, 110],
                    "currency": "RUB", "regular_price": 49.50, "promo_flag": True}) + "\n",
        encoding="utf-8",
    )
    out_json = tmp_path / "out.json"
    submission = build_submission(inputs, out_json=out_json)
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["version"] == "1.0"
    assert {v["video_id"] for v in data["videos"]} == {"vid_a", "vid_b"}
    total_tags = sum(len(v["tags"]) for v in data["videos"])
    assert total_tags == 2
