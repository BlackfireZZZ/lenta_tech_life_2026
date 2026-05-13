"""QR/barcode extraction for Lenta price-tag crops.

This module is deliberately dependency-light:
- OpenCV QRCodeDetector is always attempted.
- pyzbar is used when available for 1D barcodes and extra QR robustness.

The output is a ParsedTag carrying hackathon CSV fields in `extra_fields`.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl, urlparse

import numpy as np

from .types import ParsedTag

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


@dataclass
class QRCodeExtractor:
    confidence: float = 0.97

    def decode_payloads(self, image_bgr: np.ndarray) -> list[str]:
        payloads: list[str] = []
        payloads.extend(_decode_opencv_qr(image_bgr))
        payloads.extend(_decode_pyzbar(image_bgr))
        # Stable de-dup while preserving decoder order.
        seen: set[str] = set()
        out: list[str] = []
        for payload in payloads:
            payload = payload.strip()
            if payload and payload not in seen:
                seen.add(payload)
                out.append(payload)
        return out

    def extract(self, image_bgr: np.ndarray) -> ParsedTag:
        extra_fields: dict[str, Any] = {}
        for payload in self.decode_payloads(image_bgr):
            extra_fields.update(parse_qr_payload(payload))
        if not extra_fields:
            return ParsedTag(backend="qr", raw_text=None)
        if "qr_code_barcode" in extra_fields:
            # The visible barcode field and the QR barcode often represent the
            # same product GTIN; filling both increases useful CSV coverage.
            extra_fields.setdefault("barcode", extra_fields["qr_code_barcode"])
        return ParsedTag(
            backend="qr",
            raw_text=json.dumps(extra_fields, ensure_ascii=False),
            extra_fields=extra_fields,
            extra_confidences={k: self.confidence for k in extra_fields},
        )


def parse_qr_payload(payload: str) -> dict[str, Any]:
    """Parse common Lenta QR payload shapes into hackathon CSV fields."""
    raw = payload.strip()
    if not raw:
        return {}
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


def _decode_opencv_qr(image_bgr: np.ndarray) -> list[str]:
    try:
        import cv2  # type: ignore
    except ImportError:
        return []
    detector = cv2.QRCodeDetector()
    out: list[str] = []
    try:
        ok, decoded, _, _ = detector.detectAndDecodeMulti(image_bgr)
        if ok:
            out.extend([s for s in decoded if s])
    except Exception:
        LOGGER.debug("OpenCV detectAndDecodeMulti failed", exc_info=True)
    try:
        decoded, _, _ = detector.detectAndDecode(image_bgr)
        if decoded:
            out.append(decoded)
    except Exception:
        LOGGER.debug("OpenCV detectAndDecode failed", exc_info=True)
    return out


def _decode_pyzbar(image_bgr: np.ndarray) -> list[str]:
    try:
        from pyzbar.pyzbar import decode  # type: ignore
    except Exception:
        return []
    try:
        import cv2  # type: ignore
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
        return [item.data.decode("utf-8", errors="ignore") for item in decode(gray) if item.data]
    except Exception:
        LOGGER.debug("pyzbar decode failed", exc_info=True)
        return []


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
