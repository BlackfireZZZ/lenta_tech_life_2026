"""1D barcode reading — the ultimate reader (landed from the QR/barcode branch).

Per the hackathon metric the barcode is the **primary GT-matching key — P0**
(a row with no/incorrect barcode falls back to noisy timestamp+bbox matching
and may not match at all — see ``docs/index.md`` "five facts"), not just one
of 29 columns.

Reality on real Lenta tags (measured, see ``scripts/eval_qr.py``): the
"barcode" is **GS1 DataBar** (Stacked/Expanded, carrying the GTIN under
Application Identifier ``(01)``) or **Code-128**/EAN — *not* plain EAN-13 as
the docs assumed. The heavy lifting (decoder ensemble + escalating classical
preprocessing cascade + a doc-informed layout-ROI localiser) lives in
:mod:`price_tag_pipeline.recognition.qr_engine`; this module is the thin
:class:`CropDecoder` adapter that turns its 1D readings into the contract
output. The ``ШК:`` / ``Ш:`` *text* GTIN fallback stays the OCR decoder's job
(it owns text), reconciled by the aggregator — not duplicated here.

Plug-in contract (unchanged, as frozen in :mod:`.base`):

    input  : one rectified price-tag crop, BGR ``np.ndarray``
    output : ``list[RecognitionResult]``
             - nothing found            → ``[]``
             - found GTIN ``"460..."``  → one result whose
               ``parsed.extra_fields`` carries ``{"barcode": "<digits>"}`` and
               ``"qr_code_barcode"`` (same GTIN), ``extra_confidences`` set,
               ``found=True``.

Never raises on a bad crop (returns ``[]``). Does NOT change
:class:`CropDecoder` / :class:`RecognitionResult` / chain order / merge policy.
"""

from __future__ import annotations

import logging

import numpy as np

from ..types import ParsedTag
from .base import CropDecoder, RecognitionResult, parsed_is_empty
from .qr_engine import decode_crop

LOGGER = logging.getLogger(__name__)


class BarcodeDecoder(CropDecoder):
    """Second link of the recognition chain: read the 1D product barcode."""

    name = "barcode"

    def decode(self, crop_bgr: np.ndarray) -> list[RecognitionResult]:
        try:
            symbols = decode_crop(crop_bgr)
        except Exception:  # contract: never raise on a bad crop
            LOGGER.debug("barcode decode failed", exc_info=True)
            return []

        # Keep only linear/stacked product barcodes; pick the most trustworthy
        # (checksum-valid first, then decoder confidence). A *wrong* barcode
        # mis-keys the GT match, so prefer none over a checksum-failing read.
        ones = [s for s in symbols if s.kind == "1d" and s.checksum_ok is not False]
        if not ones:
            return []
        best = max(ones, key=lambda s: (s.checksum_ok is True, s.confidence))
        gtin = (best.gtin or best.text or "").strip()
        if not gtin:
            return []

        conf = float(best.confidence)
        parsed = ParsedTag(
            backend="barcode",
            raw_text=best.text,
            extra_fields={"barcode": gtin, "qr_code_barcode": gtin},
            # Same GTIN, but sourced from the 1D symbol rather than the QR `b`
            # field — give qr_code_barcode a hair less weight so a real QR
            # wins that column in the aggregator when both are present.
            extra_confidences={"barcode": conf, "qr_code_barcode": conf * 0.95},
        )
        if parsed_is_empty(parsed):
            return []
        return [
            RecognitionResult(
                parsed=parsed,
                decoder=self.name,
                confidence=conf,
                text=best.text,
                found=True,
            )
        ]
