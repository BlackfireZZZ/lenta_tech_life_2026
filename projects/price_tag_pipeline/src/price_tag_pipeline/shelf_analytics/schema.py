"""Data contracts for the optional product/facing shelf-state report."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .geometry import BBox


@dataclass(frozen=True)
class ProductFacing:
    """One visible product facing on a shelf frame/keyframe."""

    id: str
    bbox_xyxy: BBox
    confidence: float = 1.0
    row_id: str | None = None
    sku_id: str | None = None
    sku_confidence: float = 0.0
    display_name: str | None = None
    embedding: tuple[float, ...] | None = None
    group_hint: str | None = None
    frame_idx: int | None = None
    timestamp_s: float | None = None
    quality: str = "unknown"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = {
            "id": self.id,
            "bbox_xyxy": [round(float(v), 2) for v in self.bbox_xyxy],
            "confidence": round(float(self.confidence), 4),
            "row_id": self.row_id,
            "sku_id": self.sku_id,
            "sku_confidence": round(float(self.sku_confidence), 4),
            "display_name": self.display_name,
            "group_hint": self.group_hint,
            "frame_idx": self.frame_idx,
            "timestamp_s": self.timestamp_s,
            "quality": self.quality,
            "extra": dict(self.extra),
        }
        return out


@dataclass(frozen=True)
class PriceTagObservation:
    """One recognized price tag available to match against product groups."""

    id: str
    bbox_xyxy: BBox
    confidence: float = 1.0
    price: float | None = None
    loyalty_price: float | None = None
    currency: str = "RUB"
    product_name: str | None = None
    barcode: str | None = None
    row_id: str | None = None
    frame_idx: int | None = None
    timestamp_s: float | None = None
    source_track_id: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "bbox_xyxy": [round(float(v), 2) for v in self.bbox_xyxy],
            "confidence": round(float(self.confidence), 4),
            "price": None if self.price is None else round(float(self.price), 2),
            "loyalty_price": None if self.loyalty_price is None else round(float(self.loyalty_price), 2),
            "currency": self.currency,
            "product_name": self.product_name,
            "barcode": self.barcode,
            "row_id": self.row_id,
            "frame_idx": self.frame_idx,
            "timestamp_s": self.timestamp_s,
            "source_track_id": self.source_track_id,
            "extra": dict(self.extra),
        }


@dataclass(frozen=True)
class ShelfGroup:
    """A group of facings believed to be the same SKU or same unknown cluster."""

    id: str
    member_facing_ids: tuple[str, ...]
    bbox_xyxy: BBox
    facing_count: int
    row_id: str | None = None
    sku_id: str | None = None
    sku_confidence: float = 0.0
    display_name: str | None = None
    status: str = "unknown"
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "member_facing_ids": list(self.member_facing_ids),
            "bbox_xyxy": [round(float(v), 2) for v in self.bbox_xyxy],
            "facing_count": int(self.facing_count),
            "row_id": self.row_id,
            "sku_id": self.sku_id,
            "sku_confidence": round(float(self.sku_confidence), 4),
            "display_name": self.display_name,
            "status": self.status,
            "confidence": round(float(self.confidence), 4),
        }


@dataclass(frozen=True)
class TagProductRelation:
    """A price-tag to product-group association decision."""

    price_tag_id: str
    product_group_id: str | None
    score: float
    status: str
    reasons: tuple[str, ...] = ()
    candidates: tuple[tuple[str, float], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "price_tag_for_group",
            "price_tag_id": self.price_tag_id,
            "product_group_id": self.product_group_id,
            "score": round(float(self.score), 4),
            "status": self.status,
            "reasons": list(self.reasons),
            "candidates": [
                {"product_group_id": group_id, "score": round(float(score), 4)}
                for group_id, score in self.candidates
            ],
        }


@dataclass(frozen=True)
class ShelfState:
    """Shelf analytics output for one image/keyframe/video segment."""

    image_id: str
    product_facings: tuple[ProductFacing, ...]
    groups: tuple[ShelfGroup, ...]
    price_tags: tuple[PriceTagObservation, ...]
    relations: tuple[TagProductRelation, ...]
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "product_facings": [p.to_dict() for p in self.product_facings],
            "groups": [g.to_dict() for g in self.groups],
            "price_tags": [t.to_dict() for t in self.price_tags],
            "relations": [r.to_dict() for r in self.relations],
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }
