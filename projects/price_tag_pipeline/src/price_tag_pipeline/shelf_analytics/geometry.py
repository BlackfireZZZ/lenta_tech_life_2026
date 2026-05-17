"""Small geometry helpers for shelf analytics."""

from __future__ import annotations

from collections.abc import Iterable

BBox = tuple[float, float, float, float]


def box_area(box: BBox) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def union_box(boxes: Iterable[BBox]) -> BBox:
    boxes = list(boxes)
    if not boxes:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def horizontal_overlap_ratio(a: BBox, b: BBox) -> float:
    ax1, _, ax2, _ = a
    bx1, _, bx2, _ = b
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    denom = max(1.0, min(ax2 - ax1, bx2 - bx1))
    return inter / denom


def center(box: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def box_width(box: BBox) -> float:
    return max(0.0, box[2] - box[0])


def box_height(box: BBox) -> float:
    return max(0.0, box[3] - box[1])


def below_distance(group_box: BBox, tag_box: BBox) -> float:
    """Positive distance from a product group bottom to a tag center below it."""
    _, group_y = center((group_box[0], group_box[3], group_box[2], group_box[3]))
    _, tag_y = center(tag_box)
    return tag_y - group_y
