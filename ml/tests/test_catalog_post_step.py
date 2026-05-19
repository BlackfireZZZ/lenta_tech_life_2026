"""The GT-safe catalog post-step in the ML runner. Pins three invariants:

1. Byte/schema parity — reconciliation re-renders through the *producer*
   renderer, so the graded CSV stays the exact 29-column `\\n`-terminated
   contract; only `barcode`/`product_name` may change.
2. GT-safe — a *missing* name is filled from the catalog barcode; the
   `"нет"` absence sentinel is never overwritten.
3. Never fatal — no catalog mounted ⇒ the raw CSV passes through unchanged
   with status ``skipped`` (the graded run is independent of the catalog).
"""

import csv
import io

import pytest

from app import runner
from price_tag_pipeline.submission import HACK_CSV_COLUMNS, final_tags_to_csv
from price_tag_pipeline.types import FinalTag


def _tag(track_id, barcode, name):
    return FinalTag(
        track_id=track_id,
        bbox_xyxy=(1, 2, 3, 4),
        timestamp_s=1.0,
        source_frames=[1],
        product_name=name,
        extra_fields={"barcode": barcode, "color": "white"},
    )


@pytest.fixture(autouse=True)
def _reset_catalog_singleton():
    runner._catalog_index = None
    yield
    runner._catalog_index = None


def _catalog_file(tmp_path):
    # Lenta catalog format: `fullname;code`, cp1251, ; delimiter.
    p = tmp_path / "db_hack.csv"
    p.write_text(
        "fullname;code\nМолоко Тест 2.5% 1л;4601234567890\n",
        encoding="cp1251",
    )
    return p


def test_fill_missing_name_keeps_byte_format_and_sentinel(tmp_path, monkeypatch):
    monkeypatch.setenv("CATALOG_CSV", str(_catalog_file(tmp_path)))
    monkeypatch.setenv("HF_HOME", str(tmp_path))  # writable cache dir

    src = final_tags_to_csv(
        [
            _tag(1, "4601234567890", None),   # name missing → filled
            _tag(2, "0000000000000", "нет"),  # sentinel → untouched
        ],
        filename="25_2-10",
    )
    out, status = runner._apply_catalog(src)

    assert status.startswith("applied:")
    rd = list(csv.DictReader(io.StringIO(out)))
    # schema/byte parity: exact 29-col header, '\n' terminator, no '\r'.
    assert csv.DictReader(io.StringIO(out)).fieldnames == HACK_CSV_COLUMNS
    assert "\r" not in out and out.endswith("\n")
    assert rd[0]["product_name"] == "Молоко Тест 2.5% 1л"  # filled
    assert rd[0]["barcode"] == "4601234567890"
    assert rd[1]["product_name"] == "нет"                   # sentinel kept
    # an untouched technical column is byte-identical
    assert rd[0]["filename"] == "25_2-10"


def test_no_catalog_mounted_passes_through(tmp_path, monkeypatch):
    monkeypatch.setenv("CATALOG_CSV", str(tmp_path / "nope.csv"))
    src = final_tags_to_csv([_tag(1, "460", "X")], filename="f")
    out, status = runner._apply_catalog(src)
    assert (out, status) == (src, "skipped")


def test_degraded_csv_left_verbatim(tmp_path, monkeypatch):
    monkeypatch.setenv("CATALOG_CSV", str(_catalog_file(tmp_path)))
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    stub = "filename,frame_timestamp,x1,y1,x2,y2,barcode,name\n"
    out, status = runner._apply_catalog(stub)
    assert out == stub and status == "skipped:schema"
