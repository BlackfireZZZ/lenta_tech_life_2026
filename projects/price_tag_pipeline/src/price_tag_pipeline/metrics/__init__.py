"""Metrics package: detection mAP, OCR CER/WER, end-to-end field accuracy."""

from .detection import compute_detection_map
from .ocr_metrics import cer, wer
from .e2e import e2e_field_accuracy, per_field_report

__all__ = (
    "compute_detection_map",
    "cer",
    "wer",
    "e2e_field_accuracy",
    "per_field_report",
)
