"""Recognition seam — the contract every crop decoder must honor.

The pipeline does **detect → track → rectify → crop** upstream. This package
turns *one rectified crop* into structured :class:`ParsedTag` observations.

Ordered links (see :mod:`.chain`):

    QR  →  barcode  →  smart OCR

`QRDecoder` and `BarcodeDecoder` are intentionally thin here. Their *ultimate*
implementations (rotation/perspective-robust localization, multi-scale retry,
``ШК:``-text fallback) are being built on a **separate branch** and will be
dropped in behind :class:`CropDecoder` without touching this interface or the
chain merge policy. Treat this module as a frozen contract. Full rationale and
the plug-in checklist live in ``docs/recognition-pipeline.md``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..types import ParsedTag


@dataclass(frozen=True)
class RecognitionResult:
    """One decoder's reading of one crop.

    The pipeline turns each result into a :class:`TagObservation` and lets the
    aggregator vote per field. ``decoder``/``confidence``/``text`` also feed the
    optional JSONL audit trail.
    """

    parsed: ParsedTag
    decoder: str        # "qr" | "barcode" | "<ocr backend>"
    confidence: float   # [0, 1] — audit + (for OCR) parser confidence echo
    text: str           # raw payload / OCR text, for the audit trail
    found: bool          # did this decoder positively recognize anything?


class CropDecoder(ABC):
    """Decode one rectified crop into zero or more :class:`RecognitionResult`.

    Implementations must be cheap to construct, must never raise on a bad crop
    (return ``[]`` instead), and must return ``[]`` when they recognize nothing
    so the chain can fall through to the next link.
    """

    #: short, stable identifier used in configs, logs and the audit trail
    name: str = "decoder"

    @abstractmethod
    def decode(self, crop_bgr: np.ndarray) -> list[RecognitionResult]:
        raise NotImplementedError


def parsed_is_empty(parsed: ParsedTag) -> bool:
    """True when a :class:`ParsedTag` carries no usable field.

    Single source of truth for "did this decoder find anything?" — shared by
    every decoder and by the pipeline so the emptiness rule never drifts.
    """
    return (
        parsed.regular_price is None
        and parsed.loyalty_price is None
        and parsed.product_name is None
        and parsed.weight_value is None
        and parsed.price_per_unit_value is None
        and not parsed.extra_fields
    )
