"""Optional shelf analytics layer.

This package is deliberately separate from the graded price-tag CSV pipeline.
It consumes product facings plus recognized price tags and emits a shelf-state
report: groups of identical products, visible facing counts, and price-tag
relations.
"""

from .associate import (
    AssociationConfig,
    AssociationResult,
    TrackTrace,
    UnmatchedTrack,
    associate_tracks,
)
from .cards import (
    CardConfig,
    CardSet,
    build_card_set,
    regroup_missing_price_tag,
)
from .grouping import GroupingConfig, group_facings
from .matcher import MatchingConfig, match_price_tags
from .pipeline import build_shelf_state, build_shelf_state_from_pipeline_outputs, price_tag_from_final_tag
from .product_detector import product_facing_from_detection
from .schema import (
    AlertType,
    PriceTagObservation,
    ProductCard,
    ProductFacing,
    ShelfAlert,
    ShelfAudit,
    ShelfGroup,
    ShelfState,
    TagProductRelation,
)

__all__ = (
    "GroupingConfig",
    "group_facings",
    "MatchingConfig",
    "match_price_tags",
    "build_shelf_state",
    "build_shelf_state_from_pipeline_outputs",
    "price_tag_from_final_tag",
    "product_facing_from_detection",
    "PriceTagObservation",
    "ProductFacing",
    "ShelfGroup",
    "ShelfState",
    "TagProductRelation",
    # run-scale shelf-audit
    "AssociationConfig",
    "AssociationResult",
    "TrackTrace",
    "UnmatchedTrack",
    "associate_tracks",
    "CardConfig",
    "CardSet",
    "build_card_set",
    "regroup_missing_price_tag",
    "AlertType",
    "ProductCard",
    "ShelfAlert",
    "ShelfAudit",
)
