"""End-to-end metric smoke tests."""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from price_tag_pipeline.metrics.e2e import e2e_field_accuracy, per_field_report  # noqa: E402
from price_tag_pipeline.metrics.ocr_metrics import cer, wer  # noqa: E402


def test_cer_exact_match_is_zero():
    assert cer("молоко", "молоко") == 0.0


def test_cer_one_char_swap():
    # 1 substitution over a 6-char string == 1/6.
    assert abs(cer("молоко", "малоко") - (1 / 6)) < 1e-9


def test_wer_one_word_diff():
    assert wer("молоко 1 л", "молоко 2 л") == 1 / 3


def test_e2e_all_correct():
    pred = {
        "regular_price": 129.99, "loyalty_price": None,
        "product_name": "Молоко 1л", "weight_value": 1.0, "weight_unit": "л",
        "price_per_unit_value": None, "price_per_unit_unit": None,
        "promo_flag": False, "currency": "RUB",
    }
    gt = dict(pred)
    assert e2e_field_accuracy([(pred, gt)]) == 1.0


def test_e2e_one_field_wrong():
    pred = {
        "regular_price": 129.99, "loyalty_price": None,
        "product_name": "Молоко 1л", "weight_value": 1.0, "weight_unit": "л",
        "price_per_unit_value": None, "price_per_unit_unit": None,
        "promo_flag": False, "currency": "RUB",
    }
    gt = dict(pred, regular_price=89.90)  # mismatch
    assert e2e_field_accuracy([(pred, gt)]) == 0.0
    report = per_field_report([(pred, gt)])
    assert report["regular_price"] == 0.0
    assert report["product_name"] == 1.0
