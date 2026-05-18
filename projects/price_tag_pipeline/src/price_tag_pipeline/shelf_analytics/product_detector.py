"""Adapters for product-facing detector outputs.

The actual product detector is intentionally a separate artifact from the
price-tag detector. This module only adapts the existing generic Detection
contract into shelf-analytics ProductFacing records.
"""

from __future__ import annotations

from ..types import Detection
from .schema import ProductFacing


def product_facing_from_detection(
    det: Detection,
    prefix: str = "product",
    row_id: str | None = None,
    sku_id: str | None = None,
    sku_confidence: float = 0.0,
    display_name: str | None = None,
    embedding: tuple[float, ...] | None = None,
) -> ProductFacing:
    """Adapt a generic detector output into a product-facing observation."""

    track_or_frame = det.track_id if det.track_id is not None else det.frame_idx
    return ProductFacing(
        id=f"{prefix}_{track_or_frame}_{det.class_id}",
        bbox_xyxy=tuple(float(v) for v in det.bbox_xyxy),
        confidence=det.confidence,
        row_id=row_id,
        sku_id=sku_id,
        sku_confidence=sku_confidence,
        display_name=display_name,
        embedding=embedding,
        frame_idx=det.frame_idx,
        timestamp_s=det.timestamp_s,
        extra={"class_id": det.class_id, "class_name": det.class_name},
    )
