"""Group product facings into same-SKU or same-unknown clusters."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from .geometry import BBox, box_width, center, union_box
from .schema import ProductFacing, ShelfGroup


@dataclass(frozen=True)
class GroupingConfig:
    """Controls deterministic grouping of product facings."""

    embedding_similarity_threshold: float = 0.86
    max_adjacent_gap_ratio: float = 1.6
    min_known_sku_confidence: float = 0.55
    unknown_prefix: str = "unknown_group"


def group_facings(
    facings: list[ProductFacing] | tuple[ProductFacing, ...],
    cfg: GroupingConfig | None = None,
) -> list[ShelfGroup]:
    """Build product groups without needing SKU classes in the detector.

    Priority:
    1. explicit ``group_hint`` from annotation/review UI;
    2. confident ``sku_id``;
    3. embedding similarity inside the same shelf row;
    4. one unknown group per facing.
    """

    cfg = cfg or GroupingConfig()
    facings = sorted(facings, key=lambda f: ((f.row_id or ""), f.bbox_xyxy[1], f.bbox_xyxy[0], f.id))
    groups: list[list[ProductFacing]] = []

    for facing in facings:
        placed = False
        for members in groups:
            if _same_group(facing, members, cfg):
                members.append(facing)
                placed = True
                break
        if not placed:
            groups.append([facing])

    return [_make_group(i, members, cfg) for i, members in enumerate(groups, start=1)]


def _same_group(facing: ProductFacing, members: list[ProductFacing], cfg: GroupingConfig) -> bool:
    first = members[0]
    if facing.row_id != first.row_id:
        return False

    if facing.group_hint and first.group_hint:
        return facing.group_hint == first.group_hint
    if facing.group_hint or first.group_hint:
        return False

    if _known_sku(facing, cfg) and _known_sku(first, cfg):
        return facing.sku_id == first.sku_id and _is_spatially_adjacent(facing.bbox_xyxy, members, cfg)
    if facing.sku_id or first.sku_id:
        return False

    if facing.embedding is None or first.embedding is None:
        return False
    if not _is_spatially_adjacent(facing.bbox_xyxy, members, cfg):
        return False

    sims = [_cosine(facing.embedding, m.embedding) for m in members if m.embedding is not None]
    return bool(sims) and max(sims) >= cfg.embedding_similarity_threshold


def _known_sku(facing: ProductFacing, cfg: GroupingConfig) -> bool:
    return bool(facing.sku_id) and facing.sku_confidence >= cfg.min_known_sku_confidence


def _is_spatially_adjacent(box: BBox, members: list[ProductFacing], cfg: GroupingConfig) -> bool:
    group_box = union_box(m.bbox_xyxy for m in members)
    cx, cy = center(box)
    gx1, gy1, gx2, gy2 = group_box
    if not (gy1 <= cy <= gy2 or box[1] <= (gy1 + gy2) / 2.0 <= box[3]):
        return False

    avg_width = sum(max(1.0, box_width(m.bbox_xyxy)) for m in members) / len(members)
    if cx < gx1:
        gap = gx1 - box[2]
    elif cx > gx2:
        gap = box[0] - gx2
    else:
        gap = 0.0
    return gap <= avg_width * cfg.max_adjacent_gap_ratio


def _make_group(idx: int, members: list[ProductFacing], cfg: GroupingConfig) -> ShelfGroup:
    members = sorted(members, key=lambda f: (f.bbox_xyxy[0], f.id))
    bbox = union_box(m.bbox_xyxy for m in members)
    known = [m for m in members if _known_sku(m, cfg)]
    sku_id = known[0].sku_id if known else None
    display_name = _first_non_empty(m.display_name for m in members)
    row_id = members[0].row_id
    if sku_id:
        group_id = f"sku_{_safe_id(sku_id)}_{idx:03d}"
        status = "known_sku"
        sku_conf = sum(m.sku_confidence for m in known) / len(known)
        conf = min(1.0, 0.5 + 0.5 * sku_conf)
    else:
        group_id = f"{cfg.unknown_prefix}_{idx:03d}"
        status = "unknown_sku"
        sku_conf = 0.0
        conf = _embedding_group_confidence(members, cfg)

    return ShelfGroup(
        id=group_id,
        member_facing_ids=tuple(m.id for m in members),
        bbox_xyxy=bbox,
        facing_count=len(members),
        row_id=row_id,
        sku_id=sku_id,
        sku_confidence=sku_conf,
        display_name=display_name,
        status=status,
        confidence=conf,
    )


def _embedding_group_confidence(members: list[ProductFacing], cfg: GroupingConfig) -> float:
    vectors = [m.embedding for m in members if m.embedding is not None]
    if len(vectors) < 2:
        return 0.35
    sims = [_cosine(a, b) for i, a in enumerate(vectors) for b in vectors[i + 1 :]]
    if not sims:
        return 0.35
    return max(0.35, min(0.9, sum(sims) / len(sims)))


def _cosine(a: tuple[float, ...], b: tuple[float, ...] | None) -> float:
    if b is None or len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = sqrt(sum(x * x for x in a))
    nb = sqrt(sum(y * y for y in b))
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (na * nb)


def _safe_id(value: str) -> str:
    out = "".join(ch if ch.isalnum() else "_" for ch in value.strip().lower())
    return out.strip("_") or "sku"


def _first_non_empty(values) -> str | None:
    for value in values:
        if value:
            return value
    return None
