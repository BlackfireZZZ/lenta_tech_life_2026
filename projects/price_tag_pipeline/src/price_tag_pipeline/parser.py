"""Parser: raw OCR text -> structured ParsedTag.

Two entry points:
- `parse_text`: free-text OCR output from classical engines.
- `parse_vlm_json`: structured JSON from a VLM. Use this when available; the
  classical text parser is best-effort fallback.

Russian price-tag specifics handled here:
- Superscript kopecks (`129⁹⁹` → 129.99)
- Thousand-separator spaces (`1 299,99` → 1299.99)
- Comma decimals (`,` ↔ `.`)
- Currency suffix `₽` / `руб.` / `руб` / `RUB`
- Regular vs loyalty price disambiguation (cue words + magnitude fallback)
- Weight + unit extraction (`кг`, `г`, `л`, `мл`, `шт`)
- Price-per-unit (`/кг`, `/л`, `/шт`)
- Promo signals (`АКЦИЯ`, `СКИДКА`, `ПО КАРТЕ`, `-NN%`, `выгод`)
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

from .config import ParserConfig
from .types import HACK_EXTRA_FIELDS, ParsedTag, WeightUnit


# Superscript-digit -> ASCII-digit translation table.
_SUPERSCRIPT_DIGITS = str.maketrans({
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
})

# Currency markers — we will mostly see RUB in Lenta, but accept the others to be safe.
_RUB_TOKENS = ("₽", "руб", "rub", "р.", "р")

# Loyalty / promo cue words (lowercased, partial-match).
_LOYALTY_CUES = (
    "по карте", "с картой", "карта", "акция", "скидк", "выгод",
    "со скидкой", "promo", "промо", "сейчас", "только сейчас",
)
_REGULAR_CUES = (
    "обычная", "обыч", "регулярная", "без карты", "без скидки",
)
_PROMO_PERCENT_RE = re.compile(r"-\s?\d{1,2}\s?%")

# Weight + unit: matches "500 г", "0,5 л", "1.5 кг", "1кг", "200мл", "12 шт"
_WEIGHT_RE = re.compile(
    r"(?P<value>\d{1,5}(?:[.,]\d{1,3})?)\s*"
    r"(?P<unit>кг|г|гр|л|мл|шт|kg|g|l|ml|pcs)\b",
    flags=re.IGNORECASE | re.UNICODE,
)

# Price-per-unit: a price followed by a unit divider "/кг", "/л", "/шт"
_PPU_RE = re.compile(
    r"(?P<price>\d{1,5}(?:[  ]\d{3})*(?:[.,]\d{1,2})?)"
    r"\s*(?:₽|руб\.?|rub)?\s*/\s*"
    r"(?P<unit>кг|г|л|мл|шт|kg|g|l|ml|pcs)\b",
    flags=re.IGNORECASE | re.UNICODE,
)

# A bare price token, with optional ₽/руб suffix. Allows superscript kopecks once
# they have been normalized into ASCII digits by `_normalize_text`.
_PRICE_RE = re.compile(
    r"(?P<major>\d{1,6}(?:[  ]\d{3})*)"
    r"(?:[.,](?P<minor>\d{1,2}))?"
    r"\s*(?:(?P<cur>₽|руб\.?|rub|р\.?)|(?=\s|$|/))",
    flags=re.IGNORECASE | re.UNICODE,
)

# Russian-word run: at least 3 consecutive Cyrillic letters, can include digits/spaces between.
_PRODUCT_NAME_RE = re.compile(
    r"(?:[А-ЯЁа-яё][А-ЯЁа-яё\-]{2,}(?:\s+[А-ЯЁа-яё\d\-\.%,]+){0,8})",
    flags=re.UNICODE,
)
_BARCODE_RE = re.compile(r"(?<!\d)(\d{8,14})(?!\d)")


def _normalize_text(text: str) -> str:
    """Canonicalize OCR text for parsing.

    - NFC normalize.
    - Translate superscript digits to ASCII. We treat any contiguous superscript
      run as the *kopecks* tail of the immediately preceding integer.
    - Strip control characters and collapse whitespace.
    """
    if not text:
        return ""
    t = unicodedata.normalize("NFC", text)

    # Replace contiguous superscript-digit runs that immediately follow an integer
    # with "<int>.<kopecks>". E.g. "129⁹⁹" -> "129.99".
    def _supersub(m: re.Match) -> str:
        major = m.group(1)
        minor = m.group(2).translate(_SUPERSCRIPT_DIGITS)
        # Take at most 2 digits as kopecks; ignore the rest.
        minor = minor[:2].ljust(2, "0") if len(minor) >= 1 else "00"
        return f"{major}.{minor}"

    t = re.sub(r"(\d+)([⁰¹²³⁴⁵⁶⁷⁸⁹]{1,3})", _supersub, t)
    # Any remaining loose superscript digits become plain digits.
    t = t.translate(_SUPERSCRIPT_DIGITS)

    # Drop common nuisance chars.
    t = t.replace("​", "").replace("‌", "").replace("‍", "")
    # Collapse all kinds of whitespace.
    t = re.sub(r"[\t\r\n\f\v]+", " ", t)
    t = re.sub(r"  +", " ", t).strip()
    return t


def _parse_price_token(major_s: str, minor_s: Optional[str]) -> Optional[float]:
    """Convert a regex-matched price token into a float."""
    major = re.sub(r"[  ]", "", major_s)
    if not major.isdigit():
        return None
    if minor_s is None:
        return float(major)
    # Pad single-digit kopecks (e.g. "129.9" -> 129.90).
    minor = minor_s if len(minor_s) == 2 else (minor_s + "0")
    try:
        return float(f"{major}.{minor[:2]}")
    except ValueError:
        return None


def _find_all_prices(text: str) -> list[tuple[float, int, int]]:
    """Return list of (value, span_start, span_end) for every plausible price."""
    out: list[tuple[float, int, int]] = []
    for m in _PRICE_RE.finditer(text):
        value = _parse_price_token(m.group("major"), m.group("minor"))
        if value is None:
            continue
        out.append((value, m.start(), m.end()))
    return out


def _detect_promo(text_lower: str) -> bool:
    if _PROMO_PERCENT_RE.search(text_lower):
        return True
    return any(cue in text_lower for cue in _LOYALTY_CUES)


def _classify_two_prices(
    text_lower: str,
    prices: list[tuple[float, int, int]],
) -> tuple[Optional[float], Optional[float]]:
    """Assign (regular, loyalty) from a sorted-by-position price list.

    Strategy:
    1. If any cue word appears, attach the *closest* price by character distance
       to the matching role.
    2. Otherwise: the larger value is regular, the smaller is loyalty.
    3. If only one price exists: regular by default unless promo signals dominate.
    """
    if not prices:
        return None, None
    if len(prices) == 1:
        v = prices[0][0]
        if _detect_promo(text_lower):
            return None, v
        return v, None

    # Two or more — work with the top 2 by magnitude span (most prominent ones).
    # Heuristic: typical tags show the loyalty price largest; classical OCR doesn't
    # know that, but we can still rely on magnitude.
    largest = sorted(prices, key=lambda x: x[0], reverse=True)[:2]

    # Look for explicit cue words and attach the price that appears AFTER
    # the cue (cues like "По карте" usually immediately precede the price).
    regular_assigned: Optional[float] = None
    loyalty_assigned: Optional[float] = None

    def _price_after(cue_end: int) -> Optional[float]:
        after = [p for p in prices if p[1] >= cue_end]
        if after:
            return min(after, key=lambda x: x[1] - cue_end)[0]
        return min(prices, key=lambda x: abs(x[1] - cue_end))[0] if prices else None

    for cue in _LOYALTY_CUES:
        idx = text_lower.find(cue)
        if idx == -1:
            continue
        loyalty_assigned = _price_after(idx + len(cue))
        break
    for cue in _REGULAR_CUES:
        idx = text_lower.find(cue)
        if idx == -1:
            continue
        regular_assigned = _price_after(idx + len(cue))
        break

    if regular_assigned is not None and loyalty_assigned is not None:
        return regular_assigned, loyalty_assigned

    # Fall back to magnitude: bigger = regular, smaller = loyalty.
    big = max(largest, key=lambda x: x[0])[0]
    small = min(largest, key=lambda x: x[0])[0]
    if regular_assigned is None and loyalty_assigned is None:
        return big, small
    if regular_assigned is None:
        return (big if big != loyalty_assigned else small), loyalty_assigned
    return regular_assigned, (small if small != regular_assigned else big)


def _extract_weight(text: str) -> tuple[Optional[float], Optional[WeightUnit]]:
    m = _WEIGHT_RE.search(text)
    if not m:
        return None, None
    value_s = m.group("value").replace(",", ".")
    try:
        value = float(value_s)
    except ValueError:
        return None, None
    unit = WeightUnit.from_raw(m.group("unit"))
    return value, unit


def _extract_ppu(text: str) -> tuple[Optional[float], Optional[str]]:
    m = _PPU_RE.search(text)
    if not m:
        return None, None
    price = _parse_price_token(m.group("price"), None)
    # Re-parse with possible decimal.
    raw = m.group("price").replace(",", ".").replace(" ", "").replace(" ", "")
    try:
        price = float(raw)
    except ValueError:
        price = None
    unit = m.group("unit").lower()
    norm_unit = unit  # keep as written; normalization optional
    return price, f"руб/{norm_unit}"


def _extract_product_name(text: str) -> Optional[str]:
    """Best-effort product-name extraction.

    Picks the longest Cyrillic phrase. Drops short fragments. Trims trailing
    cue/promo words. VLM path is preferred; this is fallback.
    """
    candidates: list[str] = []
    for m in _PRODUCT_NAME_RE.finditer(text):
        s = m.group(0).strip()
        # Drop anything that is dominantly a cue word.
        if any(cue in s.lower() for cue in _LOYALTY_CUES + _REGULAR_CUES):
            continue
        if len(s) >= 5:
            candidates.append(s)
    if not candidates:
        return None
    candidates.sort(key=len, reverse=True)
    return candidates[0]


def _extract_barcode(text: str) -> Optional[str]:
    for m in _BARCODE_RE.finditer(text):
        value = m.group(1)
        # Price-tag QR payloads and OCR often contain dates; prefer GTIN-like lengths.
        if len(value) in {8, 12, 13, 14}:
            return value
    return None


def _lenient_json_loads(raw: str) -> Optional[dict]:
    """Parse VLM 'JSON' that is often slightly malformed.

    Small OCR-VLMs routinely emit: ```json fences, unquoted keys
    (``product_name: "x"``), trailing commas, smart quotes, prose around the
    object. Strict ``json.loads`` then throws away an otherwise-correct read,
    so the chain silently degrades to the crude text parser. Recover instead.
    """
    if not raw:
        return None
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s.strip())
    i, j = s.find("{"), s.rfind("}")
    if i == -1 or j == -1 or j <= i:
        return None
    s = s[i : j + 1]
    s = s.translate({0x201C: 34, 0x201D: 34, 0x2018: 39, 0x2019: 39})
    for attempt in range(4):
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else None
        except (json.JSONDecodeError, TypeError):
            if attempt == 0:  # quote bare keys
                s = re.sub(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)', r'\1"\2"\3', s)
            elif attempt == 1:  # drop trailing commas
                s = re.sub(r',(\s*[}\]])', r'\1', s)
            elif attempt == 2:  # single- to double-quoted values
                s = re.sub(r":\s*'([^']*)'", r': "\1"', s)
            else:
                break
    # Last resort: scrape "key": value pairs (string | number | null).
    out: dict[str, object] = {}
    for m in re.finditer(
        r'["\']?([A-Za-z_][A-Za-z0-9_]*)["\']?\s*:\s*'
        r'(?:"([^"]*)"|\'([^\']*)\'|(-?\d+(?:\.\d+)?)|(null|true|false))',
        s,
    ):
        key = m.group(1)
        if m.group(2) is not None:
            out[key] = m.group(2)
        elif m.group(3) is not None:
            out[key] = m.group(3)
        elif m.group(4) is not None:
            out[key] = float(m.group(4)) if "." in m.group(4) else int(m.group(4))
        else:
            out[key] = {"null": None, "true": True, "false": False}[m.group(5)]
    return out or None


_ABSENT_TOKENS = {"нет", "null", "none", "n/a", "na", "-", "—", "отсутствует"}
_BARCODE_KEYS = {"barcode", "qr_code_barcode"}

# additional_info is ONLY genuine extra text (price-tag-guide §10 row
# `additional_info`): sweetness in a rounded box, scale number, promo date
# range, threshold "от/до/при покупке N", "3=2"/"купи N". Price labels
# («без карты», «с картой», «₽/шт») and price numbers are NEVER
# additional_info — the VLM routinely dumps them there. Keep only on signal.
_AINFO_SIGNAL = (
    "сух", "сладк", "полусл", "брют", "номер на вес", "акция действ",
    "при покупке", "удачная упаков", "=", "купи", "плати",
    " от ", " до ",  # threshold "от N шт/кг" / "до N кг"
)
_AINFO_JUNK = (
    "без карт", "с карт", "по карт", "₽", "руб", "/шт", "/кг", "p/", "р/",
    "цена", "цены",
)
_DATE_RE = re.compile(r"\b\d{2}\.\d{2}\.\d{4}\b")


def _norm_extra_field(key: str, value: object) -> object:
    """Normalize one HACK extra field straight off the VLM JSON.

    - empty/absent tokens collapse to "нет" (task §5.3: absent → "нет";
      the VLM is prompted to say "нет", but also emits null/""/"-").
    - barcode / qr_code_barcode: keep digits only — the model reads the
      number correctly but with grouping spaces ("4 607124 143901"); the
      graded scorer compares barcodes as exact strings, so a perfect read
      otherwise scores 0. Drop reads shorter than a plausible GTIN.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    if s.lower() in _ABSENT_TOKENS:
        return "нет"
    if key in _BARCODE_KEYS:
        digits = re.sub(r"\D", "", s)
        # EAN-8 (8 digits) is the shortest real GTIN — a 6–7 digit read can
        # be no valid barcode. barcode is the P0 GT-matching key, so a
        # spurious short value is worse than empty (it mis-keys the match
        # AND scores 0 on the field). Matches aggregator._tag_barcode.
        return digits if len(digits) >= 8 else None
    if key == "price_discount" and "%" in s:
        # A percent is discount_amount, not the third promo PRICE. The VLM
        # routinely duplicates "-44%" here; that is never a valid price.
        return "нет"
    if key == "additional_info":
        low = s.lower()
        has_signal = _DATE_RE.search(s) is not None or any(t in low for t in _AINFO_SIGNAL)
        if has_signal:
            return s
        # No genuine signal: if it's price-label junk or just digits/price
        # tokens, it is not additional_info → absent.
        if any(t in low for t in _AINFO_JUNK) or re.fullmatch(r"[\d\s.,:;%/-]+", s):
            return "нет"
        return s
    return s


@dataclass(frozen=True)
class TagParser:
    cfg: ParserConfig

    def parse_text(self, text: str, ocr_confidence: float, backend: str = "ocr") -> ParsedTag:
        """Parse free-text OCR output. Always returns a ParsedTag (possibly empty)."""
        normalized = _normalize_text(text)
        if not normalized:
            return ParsedTag(backend=backend, raw_text=text)
        text_lower = normalized.lower()

        prices = _find_all_prices(normalized)
        # Filter to plausible price range.
        prices = [
            (v, a, b) for (v, a, b) in prices
            if self.cfg.min_price <= v <= self.cfg.max_price
        ]
        regular, loyalty = _classify_two_prices(text_lower, prices)

        weight_value, weight_unit = _extract_weight(normalized)
        ppu_value, ppu_unit = _extract_ppu(normalized)
        name = _extract_product_name(normalized)
        promo = _detect_promo(text_lower)
        barcode = _extract_barcode(normalized)
        extra_fields = {}
        extra_confidences = {}
        if barcode:
            extra_fields["barcode"] = barcode
            extra_confidences["barcode"] = ocr_confidence * 0.65

        # Confidence: scale OCR confidence by parser certainty proxies.
        price_conf = ocr_confidence * (1.0 if any(t in text_lower for t in _RUB_TOKENS) else 0.85)
        weight_conf = ocr_confidence * 0.9 if weight_value is not None else 0.0
        ppu_conf = ocr_confidence * 0.85 if ppu_value is not None else 0.0
        name_conf = ocr_confidence * 0.6 if name else 0.0

        return ParsedTag(
            regular_price=regular,
            regular_price_confidence=price_conf if regular is not None else 0.0,
            loyalty_price=loyalty,
            loyalty_price_confidence=price_conf if loyalty is not None else 0.0,
            product_name=name,
            product_name_confidence=name_conf,
            weight_value=weight_value,
            weight_unit=weight_unit,
            weight_confidence=weight_conf,
            price_per_unit_value=ppu_value,
            price_per_unit_unit=ppu_unit,
            price_per_unit_confidence=ppu_conf,
            promo_flag=promo,
            currency="RUB",
            backend=backend,
            raw_text=text,
            extra_fields=extra_fields,
            extra_confidences=extra_confidences,
        )

    def parse_vlm_json(self, raw_json: str, vlm_confidence: float, backend: str = "vlm") -> ParsedTag:
        """Parse a JSON string from a VLM. Falls back to text parsing on failure."""
        obj = _lenient_json_loads(raw_json)
        if not isinstance(obj, dict):
            return self.parse_text(raw_json, ocr_confidence=vlm_confidence, backend=backend)

        def _f(key: str) -> Optional[float]:
            v = obj.get(key)
            if v is None:
                return None
            try:
                # Allow both "129.99" and 129.99
                return float(str(v).replace(",", "."))
            except (ValueError, TypeError):
                return None

        regular = _f("regular_price")
        loyalty = _f("loyalty_price")
        weight_value = _f("weight_value")
        weight_unit = WeightUnit.from_raw(obj.get("weight_unit"))
        ppu_value = _f("price_per_unit_value")
        ppu_unit = obj.get("price_per_unit_unit")
        name = obj.get("product_name")
        promo = bool(obj.get("promo_flag", False))
        currency = obj.get("currency") or "RUB"
        extra_fields: dict[str, object] = {}
        extra_confidences: dict[str, float] = {}
        for key in HACK_EXTRA_FIELDS:
            value = _norm_extra_field(key, obj.get(key))
            if value in (None, ""):
                continue
            extra_fields[key] = value
            extra_confidences[key] = vlm_confidence

        # price_discount is a PRICE; if the VLM echoed discount_amount into it
        # (same token), it is not a third promo price → absent.
        pd = extra_fields.get("price_discount")
        if pd is not None and pd != "нет" and str(pd) == str(extra_fields.get("discount_amount", "")):
            extra_fields["price_discount"] = "нет"

        return ParsedTag(
            regular_price=regular,
            regular_price_confidence=vlm_confidence if regular is not None else 0.0,
            loyalty_price=loyalty,
            loyalty_price_confidence=vlm_confidence if loyalty is not None else 0.0,
            product_name=str(name) if name else None,
            product_name_confidence=vlm_confidence if name else 0.0,
            weight_value=weight_value,
            weight_unit=weight_unit,
            weight_confidence=vlm_confidence if weight_value is not None else 0.0,
            price_per_unit_value=ppu_value,
            price_per_unit_unit=str(ppu_unit) if ppu_unit else None,
            price_per_unit_confidence=vlm_confidence if ppu_value is not None else 0.0,
            promo_flag=promo,
            currency=currency,
            backend=backend,
            raw_text=raw_json,
            extra_fields=extra_fields,
            extra_confidences=extra_confidences,
        )
