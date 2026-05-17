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
    HACK_CSV_COLUMNS, final_tags_to_csv, hack_row_from_tag_dict,
)
from price_tag_pipeline.types import FinalTag  # noqa: E402


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


# --- graded 29-column CSV (single source of truth) -------------------------

import csv      # noqa: E402
import io       # noqa: E402


def _sample_tag() -> FinalTag:
    return FinalTag(
        track_id=7,
        bbox_xyxy=(412, 221, 520, 278),
        timestamp_s=12.43,            # -> 12430 ms
        source_frames=[372, 376],
        regular_price=129.9,
        loyalty_price=99.99,
        product_name="Молоко, 2.5%",  # comma -> must be quoted
        extra_fields={"barcode": "4601234567890", "color": "нет"},
    )


def test_hack_csv_has_29_columns_in_spec_order():
    assert len(HACK_CSV_COLUMNS) == 29
    assert HACK_CSV_COLUMNS[0] == "filename"
    assert HACK_CSV_COLUMNS[13] == "frame_timestamp"
    assert HACK_CSV_COLUMNS[-1] == "action_code_qr"


def test_final_tags_to_csv_round_trips():
    text = final_tags_to_csv([_sample_tag()], filename="25_2-10")
    # Byte-format parity with backend/app/jobs_mock.py:build_csv — the
    # gateway serves this verbatim, so \n (not \r\n) and a trailing newline.
    assert "\r\n" not in text and text.endswith("\n")
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0] == HACK_CSV_COLUMNS          # header
    assert len(rows) == 2                        # header + 1 data row
    row = dict(zip(HACK_CSV_COLUMNS, rows[1]))
    assert row["filename"] == "25_2-10"
    assert row["product_name"] == "Молоко, 2.5%"  # comma survived quoting
    assert row["price_default"] == "129.90"       # .2f, '.' decimal
    assert row["price_card"] == "99.99"
    assert row["price_discount"] == "29.91"       # derived: default - card
    assert row["barcode"] == "4601234567890"
    assert row["frame_timestamp"] == "12430"      # ms, integer string
    assert row["color"] == "нет"                  # passthrough, not empty
    assert row["id_sku"] == ""                    # unrecognized -> empty


def test_final_tags_to_csv_matches_jsonl_path():
    """The live-FinalTag path and the JSONL-replay path are one mapping."""
    tag = _sample_tag()
    via_obj = final_tags_to_csv([tag], filename="v")
    via_dict = hack_row_from_tag_dict(tag.to_dict(), filename="v")
    body = list(csv.DictReader(io.StringIO(via_obj)))[0]
    assert body == via_dict
