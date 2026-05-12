"""Parser tests focused on Russian price-tag quirks."""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from price_tag_pipeline.config import ParserConfig  # noqa: E402
from price_tag_pipeline.parser import TagParser, _normalize_text  # noqa: E402
from price_tag_pipeline.types import WeightUnit  # noqa: E402


def _parser() -> TagParser:
    return TagParser(ParserConfig(min_price=0.01, max_price=999999.99))


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def test_superscript_kopecks_become_dot_decimal():
    assert _normalize_text("129⁹⁹") == "129.99"
    assert _normalize_text("1 299⁵⁰") == "1 299.50"


def test_collapses_whitespace():
    assert _normalize_text("  hello\t\nworld  ") == "hello world"


# ---------------------------------------------------------------------------
# Single-price tags
# ---------------------------------------------------------------------------

def test_parses_simple_ruble_price_with_comma():
    out = _parser().parse_text("Цена 129,99 ₽", ocr_confidence=0.9)
    assert out.regular_price == 129.99
    assert out.loyalty_price is None
    assert out.currency == "RUB"


def test_parses_thousand_separator_space():
    out = _parser().parse_text("1 299,99 руб.", ocr_confidence=0.9)
    assert out.regular_price == 1299.99


def test_parses_superscript_kopecks_in_full_tag():
    out = _parser().parse_text("Молоко 89⁹⁰ руб", ocr_confidence=0.9)
    assert out.regular_price == 89.90


# ---------------------------------------------------------------------------
# Two-price tags: regular + loyalty
# ---------------------------------------------------------------------------

def test_two_prices_with_loyalty_cue():
    out = _parser().parse_text("Обычная 199,99 ₽\nПо карте 149,99 ₽", ocr_confidence=0.92)
    assert out.regular_price == 199.99
    assert out.loyalty_price == 149.99
    assert out.promo_flag is True


def test_two_prices_without_cue_falls_back_to_magnitude():
    out = _parser().parse_text("199,99 149,99", ocr_confidence=0.92)
    assert out.regular_price == 199.99
    assert out.loyalty_price == 149.99


def test_promo_percent_sets_promo_flag():
    out = _parser().parse_text("-30% 89,90 ₽", ocr_confidence=0.9)
    assert out.promo_flag is True


# ---------------------------------------------------------------------------
# Weight / unit
# ---------------------------------------------------------------------------

def test_weight_kg():
    out = _parser().parse_text("Сахар 1 кг 89,90 ₽", ocr_confidence=0.9)
    assert out.weight_value == 1.0
    assert out.weight_unit == WeightUnit.KG


def test_weight_g():
    out = _parser().parse_text("Хлеб 400 г 49,90 ₽", ocr_confidence=0.9)
    assert out.weight_value == 400.0
    assert out.weight_unit == WeightUnit.G


def test_weight_l_with_comma_decimal():
    out = _parser().parse_text("Сок 1,5 л 129,90 ₽", ocr_confidence=0.9)
    assert out.weight_value == 1.5
    assert out.weight_unit == WeightUnit.L


# ---------------------------------------------------------------------------
# VLM JSON path
# ---------------------------------------------------------------------------

def test_vlm_json_parse_happy_path():
    raw = '{"regular_price":"199.99","loyalty_price":"149.99","product_name":"Молоко","weight_value":1,"weight_unit":"л","promo_flag":true,"currency":"RUB"}'
    out = _parser().parse_vlm_json(raw, vlm_confidence=0.95)
    assert out.regular_price == 199.99
    assert out.loyalty_price == 149.99
    assert out.product_name == "Молоко"
    assert out.weight_unit == WeightUnit.L
    assert out.promo_flag is True


def test_vlm_invalid_json_falls_back_to_text_parser():
    out = _parser().parse_vlm_json("not json — but 99,90 ₽ visible", vlm_confidence=0.7)
    assert out.regular_price == 99.90
