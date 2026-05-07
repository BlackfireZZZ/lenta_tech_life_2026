from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .config import ParserConfig
from .types import ParsedPrice

_CURRENCY_RE = r"(?:₽|руб\.?|rub|€|\$|eur|usd)"
_UNIT_RE = r"(?:кг|kg|л|l|шт|pcs)"
_PRICE_RE = re.compile(
    rf"(?P<prefix>{_CURRENCY_RE})?\s*"
    r"(?P<major>\d{1,5})(?:[.,](?P<minor>\d{2}))?"
    rf"\s*(?P<suffix>{_CURRENCY_RE})?"
    rf"(?:\s*/\s*(?P<unit>{_UNIT_RE}))?",
    flags=re.IGNORECASE,
)


def _normalize_currency(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    s = raw.lower().replace(".", "")
    if s in {"₽", "руб", "rub"}:
        return "RUB"
    if s in {"€", "eur"}:
        return "EUR"
    if s in {"$", "usd"}:
        return "USD"
    return None


def _normalize_unit(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    s = raw.lower()
    if s in {"кг", "kg"}:
        return "kg"
    if s in {"л", "l"}:
        return "l"
    if s in {"шт", "pcs"}:
        return "pcs"
    return None


@dataclass(frozen=True)
class PriceParser:
    cfg: ParserConfig

    def parse(self, text: str, ocr_confidence: float) -> Optional[ParsedPrice]:
        best: Optional[ParsedPrice] = None
        for m in _PRICE_RE.finditer(text):
            major_s = m.group("major")
            minor_s = m.group("minor")
            value_s = major_s if minor_s is None else f"{major_s}.{minor_s}"
            value = float(value_s)
            if value < self.cfg.min_price or value > self.cfg.max_price:
                continue

            has_minor = minor_s is not None
            prior = 1.0 if (has_minor and self.cfg.decimal_preferred) else 0.82
            parsed = ParsedPrice(
                raw=m.group(0).strip(),
                normalized_price=f"{value:.2f}",
                value=value,
                currency=_normalize_currency(m.group("prefix") or m.group("suffix")),
                unit=_normalize_unit(m.group("unit")),
                confidence=max(0.0, min(1.0, ocr_confidence * prior)),
            )
            if best is None or parsed.confidence > best.confidence:
                best = parsed
        return best

