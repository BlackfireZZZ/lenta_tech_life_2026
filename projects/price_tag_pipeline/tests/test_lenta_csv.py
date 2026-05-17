"""Pure-function tests for the Lenta CSV adapter.

These do not need OpenCV (unlike test_data_pipeline.py), so they run even on
Python builds without an opencv wheel — which is exactly where the
milliseconds-vs-frame-index regression would otherwise slip through.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from price_tag_pipeline.data.loaders import (  # noqa: E402
    _parse_decimal,
    _parse_lenta_frame_idx,
    _read_lenta_csv,
)


# --- frame_timestamp is milliseconds, converted via fps -------------------

def test_frame_idx_is_ms_based() -> None:
    # 25 fps, 1000-frame clip. 200 ms -> frame 5, 6822 ms -> frame ~170.
    assert _parse_lenta_frame_idx("200", fps=25.0, frame_count=1000) == 5
    assert _parse_lenta_frame_idx("6822", fps=25.0, frame_count=1000) == 171
    assert _parse_lenta_frame_idx("0", fps=25.0, frame_count=1000) == 0


def test_small_ms_value_is_not_treated_as_frame_index() -> None:
    # Regression: 233 ms at 30 fps is frame 7, NOT frame 233 (which the old
    # frame-index-first heuristic produced because 233 < frame_count).
    assert _parse_lenta_frame_idx("233", fps=30.0, frame_count=750) == 7
    assert _parse_lenta_frame_idx("433", fps=30.0, frame_count=750) == 13


def test_frame_idx_decimal_comma_and_clamping() -> None:
    assert _parse_lenta_frame_idx("1500,0", fps=30.0, frame_count=1000) == 45
    # Annotation a hair past the last frame clamps instead of overflowing.
    assert _parse_lenta_frame_idx("999999", fps=30.0, frame_count=300) == 299


def test_frame_idx_none_and_fps_fallback() -> None:
    assert _parse_lenta_frame_idx("нет", fps=30.0, frame_count=100) is None
    assert _parse_lenta_frame_idx("", fps=30.0, frame_count=100) is None
    assert _parse_lenta_frame_idx(None, fps=30.0, frame_count=100) is None
    # fps unknown -> assume 30.
    assert _parse_lenta_frame_idx("1000", fps=0.0, frame_count=0) == 30


# --- header typo normalization --------------------------------------------

def test_read_lenta_csv_normalizes_truncated_header(tmp_path: Path) -> None:
    p = tmp_path / "26_12-20.csv"
    p.write_text(
        "filename,wholesale_level_1_coun,wholesale_level_1_price\n"
        "26_12-20.mp4,3,80\n",
        encoding="utf-8",
    )
    rows = _read_lenta_csv(p)
    assert rows[0]["wholesale_level_1_count"] == "3"
    assert "wholesale_level_1_coun" not in rows[0]


def test_read_lenta_csv_leaves_correct_header_untouched(tmp_path: Path) -> None:
    p = tmp_path / "49_5.csv"
    p.write_text(
        "filename,wholesale_level_1_count\n49_5.mp4,5\n", encoding="utf-8"
    )
    rows = _read_lenta_csv(p)
    assert rows[0]["wholesale_level_1_count"] == "5"


# --- decimal parsing sanity ----------------------------------------------

def test_parse_decimal_handles_locale_quirks() -> None:
    assert _parse_decimal("252,63") == 252.63
    assert _parse_decimal("3789.49") == 3789.49
    assert _parse_decimal("1\xa0234,56") == 1234.56
    assert _parse_decimal("нет") is None
    assert _parse_decimal("") is None
