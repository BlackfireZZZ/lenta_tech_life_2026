"""High-level shelf-state assembly helpers."""

from __future__ import annotations

from ..types import Detection, FinalTag
from .grouping import GroupingConfig, group_facings
from .matcher import MatchingConfig, match_price_tags
from .product_detector import product_facing_from_detection
from .schema import PriceTagObservation, ProductFacing, ShelfState


def build_shelf_state(
    image_id: str,
    product_facings: list[ProductFacing] | tuple[ProductFacing, ...],
    price_tags: list[PriceTagObservation] | tuple[PriceTagObservation, ...],
    grouping: GroupingConfig | None = None,
    matching: MatchingConfig | None = None,
    metadata: dict | None = None,
) -> ShelfState:
    """Build the optional shelf-state report for one keyframe/segment."""

    groups = group_facings(product_facings, grouping)
    relations = match_price_tags(groups, price_tags, matching)
    warnings: list[str] = []
    if not product_facings:
        warnings.append("no_product_facings")
    if not price_tags:
        warnings.append("no_price_tags")
    if any(r.status == "ambiguous" for r in relations):
        warnings.append("ambiguous_price_relations")

    return ShelfState(
        image_id=image_id,
        product_facings=tuple(product_facings),
        groups=tuple(groups),
        price_tags=tuple(price_tags),
        relations=tuple(relations),
        warnings=tuple(warnings),
        metadata=metadata or {},
    )


def build_shelf_state_from_pipeline_outputs(
    image_id: str,
    product_detections: list[Detection] | tuple[Detection, ...],
    price_tag_finals: list[FinalTag] | tuple[FinalTag, ...],
    grouping: GroupingConfig | None = None,
    matching: MatchingConfig | None = None,
    metadata: dict | None = None,
) -> ShelfState:
    """Connect existing detector/FinalTag outputs into the shelf-state layer.

    SKU ids, embeddings, and row ids are optional enrichment fields. Without
    them the report still exposes generic visible facings and conservative
    price-tag relations.
    """

    facings = [product_facing_from_detection(det) for det in product_detections]
    tags = [price_tag_from_final_tag(tag) for tag in price_tag_finals]
    return build_shelf_state(
        image_id=image_id,
        product_facings=facings,
        price_tags=tags,
        grouping=grouping,
        matching=matching,
        metadata=metadata,
    )


def price_tag_from_final_tag(tag: FinalTag, prefix: str = "tag") -> PriceTagObservation:
    """Adapt the existing price-tag pipeline output to shelf analytics."""

    barcode = tag.extra_fields.get("barcode") or tag.extra_fields.get("qr_code_barcode")
    return PriceTagObservation(
        id=f"{prefix}_{tag.track_id}",
        bbox_xyxy=tuple(float(v) for v in tag.bbox_xyxy),
        confidence=tag.overall_confidence,
        price=tag.regular_price,
        loyalty_price=tag.loyalty_price,
        currency=tag.currency,
        product_name=tag.product_name,
        barcode=str(barcode) if barcode else None,
        timestamp_s=tag.timestamp_s,
        source_track_id=tag.track_id,
        extra=dict(tag.extra_fields),
    )
