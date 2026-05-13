"""Domain types for the Lenta price-tag pipeline.

Schema target (graded fields):
    regular_price          float | None      (RUB.kk)
    loyalty_price          float | None      (RUB.kk, card price)
    product_name           str   | None      (Russian)
    weight_value           float | None
    weight_unit            'кг' | 'г' | 'л' | 'мл' | 'шт' | None
    price_per_unit_value   float | None
    price_per_unit_unit    str   | None      (free-form, e.g. 'руб/кг')
    promo_flag             bool
    currency               str               (default 'RUB')

Field names are kept Latin / snake_case for the JSON contract; values may be Cyrillic.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional

import numpy as np


HACK_EXTRA_FIELDS = (
    "price_discount",
    "barcode",
    "discount_amount",
    "id_sku",
    "print_datetime",
    "code",
    "additional_info",
    "color",
    "special_symbols",
    "qr_code_barcode",
    "price1_qr",
    "price2_qr",
    "price3_qr",
    "price4_qr",
    "wholesale_level_1_count",
    "wholesale_level_1_price",
    "wholesale_level_2_count",
    "wholesale_level_2_price",
    "action_price_qr",
    "action_code_qr",
)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class WeightUnit(str, Enum):
    KG = "кг"
    G = "г"
    L = "л"
    ML = "мл"
    PCS = "шт"

    @classmethod
    def from_raw(cls, raw: Optional[str]) -> Optional["WeightUnit"]:
        if not raw:
            return None
        s = raw.strip().lower().rstrip(".")
        mapping = {
            "кг": cls.KG, "kg": cls.KG, "килограмм": cls.KG,
            "г": cls.G, "g": cls.G, "грамм": cls.G, "гр": cls.G,
            "л": cls.L, "l": cls.L, "литр": cls.L,
            "мл": cls.ML, "ml": cls.ML, "миллилитр": cls.ML,
            "шт": cls.PCS, "pcs": cls.PCS, "штука": cls.PCS, "штук": cls.PCS,
        }
        return mapping.get(s)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Detection:
    """A single detector output for a single frame."""
    frame_idx: int
    timestamp_s: float
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None


@dataclass
class CropCandidate:
    """A rectified crop with quality metadata."""
    image: np.ndarray
    sharpness: float
    area_px: int
    bbox_xyxy: tuple[int, int, int, int]
    detection_confidence: float
    frame_idx: int
    timestamp_s: float


# ---------------------------------------------------------------------------
# OCR / VLM outputs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OCRResult:
    """Free-text OCR output (used by classical PaddleOCR / Tesseract paths)."""
    text: str
    confidence: float
    backend: str
    # Optional per-line breakdown (text, confidence, optional bbox)
    lines: tuple[tuple[str, float], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ParsedTag:
    """Structured tag fields after parsing or VLM extraction.

    Each `*_confidence` field is in [0, 1]; 0 means "no signal", 1 means "certain".
    The aggregator uses these for per-field weighted voting.
    """
    regular_price: Optional[float] = None
    regular_price_confidence: float = 0.0

    loyalty_price: Optional[float] = None
    loyalty_price_confidence: float = 0.0

    product_name: Optional[str] = None
    product_name_confidence: float = 0.0

    weight_value: Optional[float] = None
    weight_unit: Optional[WeightUnit] = None
    weight_confidence: float = 0.0

    price_per_unit_value: Optional[float] = None
    price_per_unit_unit: Optional[str] = None
    price_per_unit_confidence: float = 0.0

    promo_flag: bool = False
    currency: str = "RUB"

    backend: str = "unknown"
    raw_text: Optional[str] = None
    extra_fields: dict[str, Any] = field(default_factory=dict)
    extra_confidences: dict[str, float] = field(default_factory=dict)

    def has_any_price(self) -> bool:
        return self.regular_price is not None or self.loyalty_price is not None


# ---------------------------------------------------------------------------
# Per-frame observations and per-track buffers
# ---------------------------------------------------------------------------

@dataclass
class TagObservation:
    """One observation of one tag (one frame, one OCR/VLM call)."""
    frame_idx: int
    timestamp_s: float
    track_id: int
    bbox_xyxy: tuple[int, int, int, int]
    parsed: ParsedTag
    detection_confidence: float
    sharpness: float

    def field_weight(self, base_conf: float) -> float:
        """Voting weight for a single field given its parser confidence."""
        return (
            0.55 * base_conf
            + 0.25 * self.detection_confidence
            + 0.20 * min(1.0, self.sharpness / 200.0)
        )


@dataclass
class CropBufferEntry:
    """A crop kept in a track's buffer for later OCR-on-best-frames."""
    crop: CropCandidate
    quality_score: float


# ---------------------------------------------------------------------------
# Final output
# ---------------------------------------------------------------------------

@dataclass
class FinalTag:
    """Final per-track tag after voting + dedup. One row in the submission."""
    track_id: int
    bbox_xyxy: tuple[int, int, int, int]
    timestamp_s: float
    source_frames: list[int]

    regular_price: Optional[float] = None
    loyalty_price: Optional[float] = None
    product_name: Optional[str] = None
    weight_value: Optional[float] = None
    weight_unit: Optional[str] = None
    price_per_unit_value: Optional[float] = None
    price_per_unit_unit: Optional[str] = None
    promo_flag: bool = False
    currency: str = "RUB"

    field_confidences: dict[str, float] = field(default_factory=dict)
    extra_fields: dict[str, Any] = field(default_factory=dict)
    overall_confidence: float = 0.0
    n_observations: int = 0

    def to_dict(self) -> dict[str, Any]:
        out = {
            "track_id": int(self.track_id),
            "bbox": [int(v) for v in self.bbox_xyxy],
            "timestamp_s": round(float(self.timestamp_s), 3),
            "source_frames": list(self.source_frames),
            "regular_price": _round_or_none(self.regular_price, 2),
            "loyalty_price": _round_or_none(self.loyalty_price, 2),
            "product_name": self.product_name,
            "weight_value": _round_or_none(self.weight_value, 3),
            "weight_unit": self.weight_unit,
            "price_per_unit_value": _round_or_none(self.price_per_unit_value, 2),
            "price_per_unit_unit": self.price_per_unit_unit,
            "promo_flag": bool(self.promo_flag),
            "currency": self.currency,
            "field_confidences": {k: round(float(v), 4) for k, v in self.field_confidences.items()},
            "overall_confidence": round(float(self.overall_confidence), 4),
            "n_observations": int(self.n_observations),
        }
        for key in HACK_EXTRA_FIELDS:
            out[key] = self.extra_fields.get(key)
        return out


def _round_or_none(v: Optional[float], n: int) -> Optional[float]:
    return None if v is None else round(float(v), n)
