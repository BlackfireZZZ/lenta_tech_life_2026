"""Recognition layer: one rectified crop → structured tag observations.

Ordered chain ``QR → barcode → smart OCR`` behind a single stable seam
(:class:`CropDecoder`). The "ultimate" QR/barcode readers are built on a
separate branch and plug in here without changing this surface. Contract:
``docs/recognition-pipeline.md``.

Heavy backends (cv2, paddle, transformers) stay lazily imported inside the
engine modules; importing this package is cheap.
"""

from .barcode import BarcodeDecoder
from .base import CropDecoder, RecognitionResult, parsed_is_empty
from .chain import RecognitionChain, build_recognition_chain
from .ocr import BaseOCREngine, OCRDecoder, build_ocr_engine
from .qr import QRCodeExtractor, QRDecoder, parse_qr_payload

__all__ = (
    "CropDecoder",
    "RecognitionResult",
    "parsed_is_empty",
    "RecognitionChain",
    "build_recognition_chain",
    "QRDecoder",
    "QRCodeExtractor",
    "parse_qr_payload",
    "BarcodeDecoder",
    "OCRDecoder",
    "BaseOCREngine",
    "build_ocr_engine",
)
