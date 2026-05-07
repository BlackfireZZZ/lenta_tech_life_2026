from price_tag_pipeline.config import ParserConfig
from price_tag_pipeline.parser import PriceParser


def test_parse_decimal_rub_price():
    parser = PriceParser(ParserConfig(decimal_preferred=True, min_price=0.01, max_price=99999.99))
    out = parser.parse("Цена по карте: 129,99 ₽", ocr_confidence=0.90)
    assert out is not None
    assert out.normalized_price == "129.99"
    assert out.currency == "RUB"


def test_parse_unit_price():
    parser = PriceParser(ParserConfig(decimal_preferred=True, min_price=0.01, max_price=99999.99))
    out = parser.parse("2.49 €/kg", ocr_confidence=0.85)
    assert out is not None
    assert out.normalized_price == "2.49"
    assert out.currency == "EUR"
    assert out.unit == "kg"

