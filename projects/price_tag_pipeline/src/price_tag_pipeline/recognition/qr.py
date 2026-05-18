"""QR / DataMatrix extraction for Lenta price-tag crops.

This is the **first link** of the recognition chain (QR → barcode → OCR) and
the thin :class:`CropDecoder` adapter around the robust decode engine in
:mod:`price_tag_pipeline.recognition.qr_engine` (decoder ensemble + escalating
classical preprocessing cascade + doc-informed layout-ROI localiser; no ML).

Reality on real Lenta tags (measured — see ``scripts/eval_qr.py``): the small
square 2D code is a **DataMatrix** (often an opaque short code like
``1010500``), but some tags carry a real **QR** with a URL-query payload
``barcode=460...&price1=...&price4=...``. Both are handled here:

* structured QR  → many CSV fields via the alias table (:func:`parse_qr_payload`)
* opaque 2D code → surfaced raw under the **non-graded** ``datamatrix_raw``
  scratch key, never fabricated into a graded field

The 1D product barcode (GS1 DataBar / Code-128 / EAN) is decoded by the same
engine but emitted by :mod:`.barcode` — this module emits only 2D-derived
fields. Per-field reconciliation across QR/barcode/OCR is the aggregator's job
(see ``docs/recognition-pipeline.md``); not done here.

Keeps the frozen seam: ``QRDecoder.decode`` returns ``list[RecognitionResult]``
and never raises; ``QRCodeExtractor`` / ``parse_qr_payload`` stay the public
surface (``recognition.__init__`` and ``tests/test_qr.py`` import them).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl, urlparse

import numpy as np

from ..types import ParsedTag
from .base import CropDecoder, RecognitionResult, parsed_is_empty
from .qr_engine import decode_crop

LOGGER = logging.getLogger(__name__)


_ALIASES = {
    "barcode": "qr_code_barcode",
    "b": "qr_code_barcode",
    "gtin": "qr_code_barcode",
    "ean": "qr_code_barcode",
    "price1": "price1_qr",
    "p1": "price1_qr",
    "price2": "price2_qr",
    "p2": "price2_qr",
    "price3": "price3_qr",
    "p3": "price3_qr",
    "price4": "price4_qr",
    "p4": "price4_qr",
    "wholesalelevel1count": "wholesale_level_1_count",
    "wl1c": "wholesale_level_1_count",
    "wholesalelevel1price": "wholesale_level_1_price",
    "wl1p": "wholesale_level_1_price",
    "wholesalelevel2count": "wholesale_level_2_count",
    "wl2c": "wholesale_level_2_count",
    "wholesalelevel2price": "wholesale_level_2_price",
    "wl2p": "wholesale_level_2_price",
    "actionprice": "action_price_qr",
    "ap": "action_price_qr",
    "actioncode": "action_code_qr",
    "ac": "action_code_qr",
    "sku": "id_sku",
    "idsku": "id_sku",
    "id_sku": "id_sku",
    "printdatetime": "print_datetime",
    "printdate": "print_datetime",
    "code": "code",
}

_PRICE_FIELDS = {
    "price1_qr",
    "price2_qr",
    "price3_qr",
    "price4_qr",
    "wholesale_level_1_price",
    "wholesale_level_2_price",
    "action_price_qr",
}

# A bare payload that is just GTIN-length digits is a 1D barcode value, not a
# k=v blob. EAN-8 / UPC-A(12) / EAN-13 / GTIN-14. The previous baseline ran
# such a payload through the JSON/URL/kv parser only, which silently dropped
# it (a GTIN is none of those shapes) — losing the single most important
# field. Map it straight to qr_code_barcode instead.
_GTIN_RE = re.compile(r"^\s*(\d{8}|\d{12,14})\s*$")


@dataclass
class QRCodeExtractor:
    """Decode a crop's 2D codes and turn them into a :class:`ParsedTag`.

    ``confidence`` is the ceiling for fields recovered from a structured QR
    (the chain echoes it into the audit trail and the aggregator's voting).
    """

    confidence: float = 0.97

    def decode_payloads(self, image_bgr: np.ndarray) -> list[str]:
        """Raw 2D payload strings on the crop (back-compat surface)."""
        seen: set[str] = set()
        out: list[str] = []
        for sym in decode_crop(image_bgr):
            if sym.kind != "2d":
                continue
            t = sym.text.strip()
            if t and t not in seen:
                seen.add(t)
                out.append(t)
        return out

    def extract(self, image_bgr: np.ndarray) -> ParsedTag:
        extra_fields: dict[str, Any] = {}
        extra_confidences: dict[str, float] = {}

        def _put(field: str, value: Any, conf: float) -> None:
            if value in (None, ""):
                return
            if field not in extra_fields or conf > extra_confidences.get(field, 0.0):
                extra_fields[field] = value
                extra_confidences[field] = conf

        for sym in decode_crop(image_bgr):
            if sym.kind != "2d":
                continue  # 1D barcode is the BarcodeDecoder's job
            parsed = parse_qr_payload(sym.text)
            if parsed:
                for k, v in parsed.items():
                    _put(k, v, sym.confidence * self.confidence)
            else:
                # Opaque DataMatrix (e.g. "1010500"): keep it raw under a
                # non-graded scratch key so the signal isn't lost, without
                # fabricating a graded field from a guess.
                _put("datamatrix_raw", sym.text.strip(), sym.confidence)

        if not extra_fields:
            return ParsedTag(backend="qr", raw_text=None)

        # The QR `b` field and the visible barcode are usually the same GTIN;
        # cross-filling both maximises useful CSV coverage (the aggregator
        # still reconciles against the dedicated BarcodeDecoder reading).
        if "qr_code_barcode" in extra_fields and "barcode" not in extra_fields:
            _put("barcode", extra_fields["qr_code_barcode"],
                 extra_confidences["qr_code_barcode"] * 0.9)

        return ParsedTag(
            backend="qr",
            raw_text=json.dumps(extra_fields, ensure_ascii=False),
            extra_fields=extra_fields,
            extra_confidences=extra_confidences,
        )


def parse_qr_payload(payload: str) -> dict[str, Any]:
    """Parse common Lenta QR payload shapes into hackathon CSV fields.

    Order: a bare GTIN (1D value encoded in a 2D code), then JSON, URL-query,
    then ``k=v`` / ``k:v`` blobs. Unknown keys are dropped; known keys are
    normalised through :data:`_ALIASES`.
    """
    raw = payload.strip()
    if not raw:
        return {}

    m = _GTIN_RE.match(raw)
    if m:
        return {"qr_code_barcode": m.group(1)}

    data: dict[str, Any] = {}
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            data.update(obj)
    except json.JSONDecodeError:
        pass

    if not data:
        parsed = urlparse(raw)
        query = parsed.query or raw
        pairs = parse_qsl(query, keep_blank_values=False)
        if pairs:
            data.update(dict(pairs))

    if not data:
        for part in re.split(r"[;&|\n\r]+", raw):
            if "=" in part:
                k, v = part.split("=", 1)
            elif ":" in part:
                k, v = part.split(":", 1)
            else:
                continue
            data[k.strip()] = v.strip()

    normalized: dict[str, Any] = {}
    for key, value in data.items():
        out_key = _ALIASES.get(_norm_key(key))
        if out_key is None or value in (None, ""):
            continue
        normalized[out_key] = _normalize_value(out_key, value)
    return normalized


def _norm_key(key: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).strip().lower())


def _normalize_value(field: str, value: object) -> object:
    s = str(value).strip()
    if field in _PRICE_FIELDS:
        try:
            return round(float(s.replace(",", ".").replace(" ", "").replace("\xa0", "")), 2)
        except ValueError:
            return s
    if field.endswith("_count"):
        try:
            return int(round(float(s.replace(",", "."))))
        except ValueError:
            return s
    return s


# ---------------------------------------------------------------------------
# CropDecoder adapter — the stable seam used by RecognitionChain
# ---------------------------------------------------------------------------

class QRDecoder(CropDecoder):
    """First link of the recognition chain: decode 2D payloads on a crop.

    Returns at most one :class:`RecognitionResult`. ``found`` is True only
    when a 2D payload produced at least one usable field; an unreadable /
    absent QR yields ``[]`` (the chain then relies on barcode + OCR).
    """

    name = "qr"

    def __init__(self) -> None:
        self._extractor = QRCodeExtractor()

    def decode(self, crop_bgr: np.ndarray) -> list[RecognitionResult]:
        try:
            parsed = self._extractor.extract(crop_bgr)
        except Exception:  # contract: never raise on a bad crop
            LOGGER.debug("qr decode failed", exc_info=True)
            return []
        if parsed_is_empty(parsed):
            return []
        return [
            RecognitionResult(
                parsed=parsed,
                decoder=self.name,
                confidence=self._extractor.confidence,
                text=parsed.raw_text or "",
                found=True,
            )
        ]
