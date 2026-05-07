from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Detection:
    frame_idx: int
    timestamp_s: float
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None


@dataclass(frozen=True)
class OCRResult:
    text: str
    confidence: float
    backend: str


@dataclass(frozen=True)
class ParsedPrice:
    raw: str
    normalized_price: str
    value: float
    currency: Optional[str]
    unit: Optional[str]
    confidence: float


@dataclass
class PriceObservation:
    frame_idx: int
    timestamp_s: float
    track_id: int
    bbox_xyxy: tuple[int, int, int, int]
    parsed: ParsedPrice
    detection_confidence: float
    ocr_confidence: float
    sharpness: float

    @property
    def score(self) -> float:
        return (
            0.45 * self.parsed.confidence
            + 0.25 * self.ocr_confidence
            + 0.20 * self.detection_confidence
            + 0.10 * min(1.0, self.sharpness / 200.0)
        )


@dataclass
class FinalPrediction:
    price: str
    price_value: float
    currency: Optional[str]
    unit: Optional[str]
    confidence: float
    track_id: int
    bbox_xyxy: tuple[int, int, int, int]
    timestamp_s: float
    source_frames: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "price": self.price,
            "price_value": self.price_value,
            "currency": self.currency,
            "unit": self.unit,
            "confidence": round(float(self.confidence), 4),
            "track_id": self.track_id,
            "bbox": list(self.bbox_xyxy),
            "timestamp_s": round(float(self.timestamp_s), 3),
            "source_frames": self.source_frames,
        }


@dataclass
class CropCandidate:
    image: np.ndarray
    sharpness: float
    area_px: int
    bbox_xyxy: tuple[int, int, int, int]

