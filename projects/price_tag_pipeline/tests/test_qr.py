"""QR payload parser tests."""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from price_tag_pipeline.recognition.qr import parse_qr_payload  # noqa: E402


def test_parse_qr_url_query_aliases():
    payload = "https://lenta.test/tag?b=4601234567890&p1=129.99&p2=99,99&aP=89.9&aC=A123&wL1C=3&wL1P=80"
    out = parse_qr_payload(payload)
    assert out["qr_code_barcode"] == "4601234567890"
    assert out["price1_qr"] == 129.99
    assert out["price2_qr"] == 99.99
    assert out["action_price_qr"] == 89.90
    assert out["action_code_qr"] == "A123"
    assert out["wholesale_level_1_count"] == 3
    assert out["wholesale_level_1_price"] == 80.0


def test_parse_qr_json_aliases():
    payload = '{"barcode":"4601234567890","price1":129.99,"actionCode":"PROMO42"}'
    out = parse_qr_payload(payload)
    assert out["qr_code_barcode"] == "4601234567890"
    assert out["price1_qr"] == 129.99
    assert out["action_code_qr"] == "PROMO42"
