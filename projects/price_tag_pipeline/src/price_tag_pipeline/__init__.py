"""Lenta price-tag recognition pipeline.

This top-level package is intentionally lean: importing it pulls in only
config + types, NOT the heavy modules (detector / rectifier / ocr) that
depend on cv2, ultralytics, paddleocr, etc. Import those directly when
you need them, e.g.:

    from price_tag_pipeline.pipeline import PriceTagPipeline
    from price_tag_pipeline.config import load_config
"""

from .config import PipelineConfig, load_config
from .types import (
    Detection,
    CropCandidate,
    OCRResult,
    ParsedTag,
    TagObservation,
    FinalTag,
    WeightUnit,
)

__all__ = (
    "PipelineConfig",
    "load_config",
    "Detection",
    "CropCandidate",
    "OCRResult",
    "ParsedTag",
    "TagObservation",
    "FinalTag",
    "WeightUnit",
)
