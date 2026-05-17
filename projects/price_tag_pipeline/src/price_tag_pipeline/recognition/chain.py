"""The recognition chain: ordered crop decoders + merge policy.

Pipeline order for one rectified crop::

    QR  →  barcode  →  smart OCR

Merge policy ``qr_first_fill_gaps`` (the project default):

- run **every** enabled decoder on the crop (QR is *not* a short-circuit);
- emit each decoder's reading as a separate observation.

Per-**field** merging — QR's high-confidence fields beating OCR where both
speak, OCR filling the fields QR is silent on — is delegated to the
:class:`TrackAggregator`'s existing weighted voting (QR carries confidence
~0.97; OCR carries its parser confidence). It is **deliberately not**
re-implemented here: the chain owns *which decoders run and in what order*,
the aggregator owns *field-level reconciliation*. Keeping that boundary is
what lets the separate QR/barcode branch swap decoder internals without
touching voting. Full rationale: ``docs/recognition-pipeline.md``.
"""

from __future__ import annotations

import logging

import numpy as np

from ..config import PipelineConfig
from .barcode import BarcodeDecoder
from .base import CropDecoder, RecognitionResult
from .ocr import OCRDecoder
from .qr import QRDecoder

LOGGER = logging.getLogger(__name__)


class RecognitionChain:
    """An ordered list of :class:`CropDecoder` run over each crop."""

    def __init__(self, decoders: list[CropDecoder]) -> None:
        self._decoders = list(decoders)

    @property
    def decoder_names(self) -> list[str]:
        return [d.name for d in self._decoders]

    def decode(self, crop_bgr: np.ndarray) -> list[RecognitionResult]:
        """Run the whole chain on one crop, concatenating every reading.

        Order is preserved (QR first) so audit/logs read top-down; emptiness
        filtering and turning results into observations is the pipeline's job.
        """
        out: list[RecognitionResult] = []
        for decoder in self._decoders:
            out.extend(decoder.decode(crop_bgr))
        return out


def build_recognition_chain(cfg: PipelineConfig) -> RecognitionChain:
    """Assemble the chain from config.

    Decoders are appended in the fixed QR → barcode → OCR order; each link is
    independently switchable via the ``recognition:`` config block (all on by
    default, so existing profiles without that block keep current behavior).
    """
    rc = cfg.recognition
    decoders: list[CropDecoder] = []
    if rc.enable_qr:
        decoders.append(QRDecoder())
    if rc.enable_barcode:
        decoders.append(BarcodeDecoder())
    if rc.enable_ocr:
        decoders.append(OCRDecoder.from_config(cfg.ocr, cfg.parser))
    if not decoders:
        LOGGER.warning(
            "Recognition chain is empty (all decoders disabled): no fields "
            "will ever be extracted. Check the 'recognition:' config block."
        )
    LOGGER.info(
        "Recognition chain: %s (policy=%s)",
        " -> ".join(d.name for d in decoders) or "<empty>",
        rc.merge_policy,
    )
    return RecognitionChain(decoders)
