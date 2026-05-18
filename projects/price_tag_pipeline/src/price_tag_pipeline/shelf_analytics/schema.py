"""Data contracts for the optional product/facing shelf-state report."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
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


# ---------------------------------------------------------------------------
# Run-scale shelf-audit contracts (killer feature: OOS + missing-tag + cards).
# These are video/run-scale and intentionally separate from the per-image
# ShelfState above and from the graded 29-column CSV. See docs/shelf-audit.md.
# ---------------------------------------------------------------------------


class AlertType(str, Enum):
    OUT_OF_STOCK = "OUT_OF_STOCK"            # price tag with no product above it
    MISSING_PRICE_TAG = "MISSING_PRICE_TAG"  # product with no price tag


@dataclass(frozen=True)
class ShelfAlert:
    """One actionable shelf-audit finding for the alerts feed."""

    id: str
    type: AlertType
    severity: str  # "high" | "medium"
    video_id: str
    timestamp_s: float
    frame_idx: int
    bbox_xyxy: BBox  # original-frame coords (drawable on the raw video)
    track_id: int | None = None
    evidence_crop: str | None = None
    price_tag: dict[str, Any] | None = None
    product: dict[str, Any] | None = None
    first_seen_s: float | None = None
    last_seen_s: float | None = None
    persistence_frames: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "severity": self.severity,
            "video_id": self.video_id,
            "timestamp_s": round(float(self.timestamp_s), 3),
            "frame_idx": int(self.frame_idx),
            "bbox_xyxy": [round(float(v), 2) for v in self.bbox_xyxy],
            "track_id": self.track_id,
            "evidence_crop": self.evidence_crop,
            "price_tag": self.price_tag,
            "product": self.product,
            "first_seen_s": (None if self.first_seen_s is None
                             else round(float(self.first_seen_s), 3)),
            "last_seen_s": (None if self.last_seen_s is None
                            else round(float(self.last_seen_s), 3)),
            "persistence_frames": int(self.persistence_frames),
        }


@dataclass(frozen=True)
class ProductCard:
    """One distinct product seen in the run (the UI 'card')."""

    card_id: str
    product_track_id: int
    best_crop: str | None = None
    facing_count: int = 1
    seen_from_s: float | None = None
    seen_to_s: float | None = None
    matched_price_tag_track_id: int | None = None
    price: float | None = None
    loyalty_price: float | None = None
    name: str | None = None
    barcode: str | None = None
    catalog: dict[str, Any] | None = None
    embedding_dim: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "card_id": self.card_id,
            "product_track_id": int(self.product_track_id),
            "best_crop": self.best_crop,
            "facing_count": int(self.facing_count),
            "seen_from_s": (None if self.seen_from_s is None
                            else round(float(self.seen_from_s), 3)),
            "seen_to_s": (None if self.seen_to_s is None
                          else round(float(self.seen_to_s), 3)),
            "matched_price_tag_track_id": self.matched_price_tag_track_id,
            "price": None if self.price is None else round(float(self.price), 2),
            "loyalty_price": (None if self.loyalty_price is None
                              else round(float(self.loyalty_price), 2)),
            "name": self.name,
            "barcode": self.barcode,
            "catalog": self.catalog,
            "embedding_dim": self.embedding_dim,
        }


@dataclass(frozen=True)
class ShelfAudit:
    """Full run-scale shelf-audit report for one video."""

    video_id: str
    fps: float
    summary: dict[str, Any]
    relations: tuple[TagProductRelation, ...] = ()
    alerts: tuple[ShelfAlert, ...] = ()
    cards: tuple[ProductCard, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "fps": round(float(self.fps), 3),
            "summary": dict(self.summary),
            "relations": [r.to_dict() for r in self.relations],
            "alerts": [a.to_dict() for a in self.alerts],
            "cards": [c.to_dict() for c in self.cards],
            "metadata": dict(self.metadata),
        }
