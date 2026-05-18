"""Match recognized price tags to product-facing groups."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

from .geometry import below_distance, box_height, center, horizontal_overlap_ratio
from .schema import PriceTagObservation, ShelfGroup, TagProductRelation


@dataclass(frozen=True)
class MatchingConfig:
    """Scoring knobs for price-tag to product-group matching."""

    min_score: float = 0.42
    ambiguous_margin: float = 0.08
    max_below_distance_ratio: float = 2.2
    allow_row_mismatch: bool = False


def match_price_tags(
    groups: list[ShelfGroup] | tuple[ShelfGroup, ...],
    price_tags: list[PriceTagObservation] | tuple[PriceTagObservation, ...],
    cfg: MatchingConfig | None = None,
) -> list[TagProductRelation]:
    """Assign each price tag to the most likely product group.

    This is intentionally rule-based for the MVP: it is debuggable, needs very
    little labeled data, and can later become training data for a learned
    association model.
    """

    cfg = cfg or MatchingConfig()
    relations: list[TagProductRelation] = []
    for tag in price_tags:
        scored = sorted(
            ((_score(group, tag, cfg), group) for group in groups),
            key=lambda item: item[0],
            reverse=True,
        )
        candidates = tuple((group.id, score) for score, group in scored[:3] if score > 0)
        if not scored or scored[0][0] < cfg.min_score:
            relations.append(
                TagProductRelation(
                    price_tag_id=tag.id,
                    product_group_id=None,
                    score=scored[0][0] if scored else 0.0,
                    status="ambiguous",
                    reasons=("no_candidate_above_threshold",),
                    candidates=candidates,
                )
            )
            continue

        best_score, best_group = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        if second_score > 0 and best_score - second_score < cfg.ambiguous_margin:
            relations.append(
                TagProductRelation(
                    price_tag_id=tag.id,
                    product_group_id=None,
                    score=best_score,
                    status="ambiguous",
                    reasons=("candidate_scores_too_close",),
                    candidates=candidates,
                )
            )
            continue

        relations.append(
            TagProductRelation(
                price_tag_id=tag.id,
                product_group_id=best_group.id,
                score=best_score,
                status="ok",
                reasons=_reasons(best_group, tag),
                candidates=candidates,
            )
        )
    return relations


def _score(group: ShelfGroup, tag: PriceTagObservation, cfg: MatchingConfig) -> float:
    if group.row_id and tag.row_id and group.row_id != tag.row_id and not cfg.allow_row_mismatch:
        return 0.0

    overlap = horizontal_overlap_ratio(group.bbox_xyxy, tag.bbox_xyxy)
    gx, _ = center(group.bbox_xyxy)
    tx, _ = center(tag.bbox_xyxy)
    group_width = max(1.0, group.bbox_xyxy[2] - group.bbox_xyxy[0])
    x_center_score = max(0.0, 1.0 - abs(gx - tx) / group_width)

    dy = below_distance(group.bbox_xyxy, tag.bbox_xyxy)
    tag_h = max(1.0, box_height(tag.bbox_xyxy))
    if dy < -tag_h * 0.5:
        y_score = 0.0
    else:
        max_dist = tag_h * cfg.max_below_distance_ratio
        y_score = max(0.0, 1.0 - max(0.0, dy) / max_dist)

    text_score = _text_similarity(group.display_name or group.sku_id, tag.product_name)
    row_score = 1.0 if not group.row_id or not tag.row_id or group.row_id == tag.row_id else 0.0

    return (
        0.40 * overlap
        + 0.25 * y_score
        + 0.20 * x_center_score
        + 0.10 * text_score
        + 0.05 * row_score
    )


def _text_similarity(a: str | None, b: str | None) -> float:
    if not a or not b:
        return 0.0
    a_norm = " ".join(str(a).lower().split())
    b_norm = " ".join(str(b).lower().split())
    if not a_norm or not b_norm:
        return 0.0
    if a_norm in b_norm or b_norm in a_norm:
        return 1.0
    return SequenceMatcher(a=a_norm, b=b_norm).ratio()


def _reasons(group: ShelfGroup, tag: PriceTagObservation) -> tuple[str, ...]:
    reasons: list[str] = []
    if horizontal_overlap_ratio(group.bbox_xyxy, tag.bbox_xyxy) > 0.4:
        reasons.append("horizontal_overlap")
    if below_distance(group.bbox_xyxy, tag.bbox_xyxy) >= 0:
        reasons.append("tag_below_group")
    if group.row_id and tag.row_id and group.row_id == tag.row_id:
        reasons.append("same_row")
    if _text_similarity(group.display_name or group.sku_id, tag.product_name) > 0.75:
        reasons.append("text_similarity")
    return tuple(reasons)
