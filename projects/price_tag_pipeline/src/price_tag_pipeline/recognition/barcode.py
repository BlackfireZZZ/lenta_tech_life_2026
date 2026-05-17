"""1D barcode reading — STUB / TODO (owned by a separate branch).

The ultimate barcode reader is being built on a **separate branch**:
rotation/perspective-robust EAN-13 / EAN-8 / Code-128 localization and
decoding, super-resolution retry on tiny crops, and the ``ШК:`` / ``Ш:``
*text* barcode fallback for tag layouts that print the GTIN as digits instead
of a graphical symbology (see ``docs/strategy.md`` and the price-tag guide).

It will be dropped in here behind :class:`CropDecoder` **without** changing
the interface or the chain merge policy. Until then this is a deliberate
no-op so the chain structure is complete, importable and unit-testable.

Why barcode matters: per the hackathon metric the barcode is the **primary
GT-matching key** (a row with no/incorrect barcode falls back to noisy
timestamp+bbox matching and may not match at all). It is **P0**, not just one
of 29 columns — see ``docs/index.md`` "five facts".

Plug-in contract for the separate branch (do not deviate):

    input : one rectified price-tag crop, BGR ``np.ndarray``
    output : ``list[RecognitionResult]``
             - nothing found            → ``[]``
             - found GTIN ``"460..."``  → one result whose ``parsed.extra_fields``
               carries ``{"barcode": "<digits>"}`` (also set
               ``"qr_code_barcode"`` when it is the same GTIN), with
               ``extra_confidences`` set and ``found=True``.

Do NOT change :class:`CropDecoder` / :class:`RecognitionResult` / the
``RecognitionChain`` ordering. See ``docs/recognition-pipeline.md``.
"""

from __future__ import annotations

import logging

import numpy as np

from .base import CropDecoder, RecognitionResult

LOGGER = logging.getLogger(__name__)


class BarcodeDecoder(CropDecoder):
    """Second link of the recognition chain. No-op until the reader lands."""

    name = "barcode"

    def decode(self, crop_bgr: np.ndarray) -> list[RecognitionResult]:
        # TODO(separate-branch): ultimate 1D-barcode + ``ШК:``-text reader
        # lands here. Must satisfy the plug-in contract in this module's
        # docstring. Keep returning [] until then so the chain falls through
        # to OCR and nothing emits a fake/empty barcode observation.
        return []
