"""Track-level temporal association of price tags and product facings.

The friend's :func:`shelf_analytics.matcher.match_price_tags` is single-image
and only emits tag->group. The shelf-audit killer feature needs the *inverse*
too — price tags with no product (OUT_OF_STOCK) and products with no tag
(MISSING_PRICE_TAG) — and robustness against robot motion / occlusion / partial
views. So we associate whole tracks over their shared frames.

Coordinate space (P0-verified, critical): the detector un-projects boxes to the
ORIGINAL sideways frame. The "price tag sits below its product" geometry only
holds in CCW-upright space, so every box is mapped via
:func:`geometry.to_upright` before scoring. See docs/shelf-audit.md §5/§7.

This module is pure-Python and deterministic — no detector/IO deps — so it is
fully unit-testable and stays off the graded CSV path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean

from .geometry import (
    BBox,
    below_distance,
    box_height,
    box_width,
    center,
    horizontal_overlap_ratio,
    to_upright,
)
from .schema import TagProductRelation


@dataclass(frozen=True)
class TrackTrace:
    """One tracked object's per-frame boxes in ORIGINAL-frame coords.

    ``boxes`` maps ``frame_idx -> (x1, y1, x2, y2)``. ``meta`` carries
    side-specific payload (price tag: price/name/barcode/timestamp; product:
    confidences) — opaque to the associator, passed through for the runner.
    """

    track_id: int
    boxes: dict[int, BBox]
    meta: dict = field(default_factory=dict)

    @property
    def frames(self) -> set[int]:
        return set(self.boxes)

    @property
    def n_frames(self) -> int:
        return len(self.boxes)


@dataclass(frozen=True)
class AssociationConfig:
    """Scoring + decision knobs (mirrors matcher.MatchingConfig where sensible)."""

    rotation: str = "ccw"           # frame_rotation used by the detector
    min_co_frames: int = 3          # need this many shared frames to score a pair
    min_score: float = 0.42         # below this a tag has no product (OOS candidate)
    ambiguous_margin: float = 0.08  # top-two closer than this -> ambiguous
    max_below_distance_ratio: float = 2.2   # tag may sit up to N tag-heights below
    drop_worst_frac: float = 0.20   # trim this fraction of worst co-frame scores
    persistence_min_frames: int = 10  # an unmatched track shorter than this is noise
    # geometry weights (normalised; per-frame has no text/row signal)
    w_overlap: float = 0.45
    w_below: float = 0.30
    w_xcenter: float = 0.25


@dataclass(frozen=True)
class UnmatchedTrack:
    """An unmatched price-tag or product track (candidate alert)."""

    track_id: int
    kind: str  # "price_tag" | "product"
    n_frames: int
    first_frame: int
    last_frame: int
    first_s: float
    last_s: float
    representative_bbox: BBox  # original-frame coords
    persistent: bool
    best_score: float  # best score it reached against the other side (debug)

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "kind": self.kind,
            "n_frames": self.n_frames,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "first_s": round(self.first_s, 3),
            "last_s": round(self.last_s, 3),
            "representative_bbox": [round(float(v), 2) for v in self.representative_bbox],
            "persistent": self.persistent,
            "best_score": round(self.best_score, 4),
        }


@dataclass(frozen=True)
class AssociationResult:
    relations: tuple[TagProductRelation, ...]
    unmatched_price_tags: tuple[UnmatchedTrack, ...]
    unmatched_products: tuple[UnmatchedTrack, ...]

    def to_dict(self) -> dict:
        return {
            "relations": [r.to_dict() for r in self.relations],
            "unmatched_price_tags": [u.to_dict() for u in self.unmatched_price_tags],
            "unmatched_products": [u.to_dict() for u in self.unmatched_products],
        }


def _pair_score(tag_up: BBox, prod_up: BBox, cfg: AssociationConfig) -> float:
    """Geometry score for one co-frame, in upright space. 0 = incompatible.

    Mirrors matcher._score's geometry terms (overlap / tag-below-product /
    x-centre), renormalised — per-frame association has no text or row signal.
    """
    overlap = horizontal_overlap_ratio(prod_up, tag_up)

    gx, _ = center(prod_up)
    tx, _ = center(tag_up)
    gw = max(1.0, box_width(prod_up))
    x_score = max(0.0, 1.0 - abs(gx - tx) / gw)

    dy = below_distance(prod_up, tag_up)  # +ve: tag below product (expected)
    tag_h = max(1.0, box_height(tag_up))
    if dy < -tag_h * 0.5:  # tag clearly ABOVE the product -> not its tag
        y_score = 0.0
    else:
        max_dist = tag_h * cfg.max_below_distance_ratio
        y_score = max(0.0, 1.0 - max(0.0, dy) / max_dist)

    return cfg.w_overlap * overlap + cfg.w_below * y_score + cfg.w_xcenter * x_score


def _trimmed_mean(values: list[float], drop_frac: float) -> float:
    """Mean after dropping the lowest ``drop_frac`` fraction (robust to a few
    bad co-frames where a product box flickered)."""
    if not values:
        return 0.0
    vals = sorted(values)
    drop = int(len(vals) * drop_frac)
    kept = vals[drop:] if drop < len(vals) else vals
    return fmean(kept)


def _representative_bbox(boxes: dict[int, BBox]) -> BBox:
    """Per-coordinate median box over the track (stable vs. a single jittered
    frame; original-frame coords, suitable for drawing/cropping)."""
    xs1 = sorted(b[0] for b in boxes.values())
    ys1 = sorted(b[1] for b in boxes.values())
    xs2 = sorted(b[2] for b in boxes.values())
    ys2 = sorted(b[3] for b in boxes.values())
    m = len(xs1) // 2
    return (xs1[m], ys1[m], xs2[m], ys2[m])


def _reasons(tag_up: BBox, prod_up: BBox, cfg: AssociationConfig) -> tuple[str, ...]:
    out: list[str] = []
    if horizontal_overlap_ratio(prod_up, tag_up) > 0.4:
        out.append("horizontal_overlap")
    if below_distance(prod_up, tag_up) >= 0:
        out.append("tag_below_product")
    return tuple(out)


def associate_tracks(
    tag_traces: list[TrackTrace] | tuple[TrackTrace, ...],
    product_traces: list[TrackTrace] | tuple[TrackTrace, ...],
    frame_w: int,
    frame_h: int,
    fps: float,
    cfg: AssociationConfig | None = None,
) -> AssociationResult:
    """Associate price-tag tracks with product tracks over shared frames.

    Returns ``ok``/``ambiguous`` relations plus the unmatched tag and product
    tracks (with a persistence flag) the runner turns into OUT_OF_STOCK /
    MISSING_PRICE_TAG alerts. Deterministic: inputs are sorted by track_id.
    """

    cfg = cfg or AssociationConfig()
    fps = fps if fps and fps > 1e-6 else 1.0
    tags = sorted(tag_traces, key=lambda t: t.track_id)
    prods = sorted(product_traces, key=lambda p: p.track_id)

    # Cache upright boxes per trace per frame (each box mapped once).
    def _upright(trace: TrackTrace) -> dict[int, BBox]:
        return {
            f: to_upright(b, frame_w, frame_h, cfg.rotation)
            for f, b in trace.boxes.items()
        }

    tag_up = {t.track_id: _upright(t) for t in tags}
    prod_up = {p.track_id: _upright(p) for p in prods}

    # Pairwise robust score over co-frames.
    scores: dict[int, list[tuple[float, TrackTrace]]] = {}
    best_for_product: dict[int, float] = {p.track_id: 0.0 for p in prods}
    for t in tags:
        ranked: list[tuple[float, TrackTrace]] = []
        for p in prods:
            co = sorted(t.frames & p.frames)
            if len(co) < cfg.min_co_frames:
                continue
            per = [_pair_score(tag_up[t.track_id][f], prod_up[p.track_id][f], cfg)
                   for f in co]
            s = _trimmed_mean(per, cfg.drop_worst_frac)
            if s > 0:
                ranked.append((s, p))
                best_for_product[p.track_id] = max(best_for_product[p.track_id], s)
        ranked.sort(key=lambda it: (-it[0], it[1].track_id))
        scores[t.track_id] = ranked

    relations: list[TagProductRelation] = []
    matched_product_ids: set[int] = set()
    unmatched_tag_ids: list[tuple[int, float]] = []

    for t in tags:
        ranked = scores[t.track_id]
        candidates = tuple((f"product_{p.track_id}", round(s, 4))
                           for s, p in ranked[:3])
        best = ranked[0] if ranked else None
        if best is None or best[0] < cfg.min_score:
            relations.append(TagProductRelation(
                price_tag_id=f"tag_{t.track_id}",
                product_group_id=None,
                score=best[0] if best else 0.0,
                status="ambiguous",
                reasons=("no_candidate_above_threshold",),
                candidates=candidates,
            ))
            unmatched_tag_ids.append((t.track_id, best[0] if best else 0.0))
            continue

        second = ranked[1][0] if len(ranked) > 1 else 0.0
        if second > 0 and best[0] - second < cfg.ambiguous_margin:
            relations.append(TagProductRelation(
                price_tag_id=f"tag_{t.track_id}",
                product_group_id=None,
                score=best[0],
                status="ambiguous",
                reasons=("candidate_scores_too_close",),
                candidates=candidates,
            ))
            unmatched_tag_ids.append((t.track_id, best[0]))
            continue

        s, p = best
        matched_product_ids.add(p.track_id)
        # reasons from the best shared frame
        co = sorted(t.frames & p.frames)
        bf = max(co, key=lambda f: _pair_score(
            tag_up[t.track_id][f], prod_up[p.track_id][f], cfg))
        relations.append(TagProductRelation(
            price_tag_id=f"tag_{t.track_id}",
            product_group_id=f"product_{p.track_id}",
            score=s,
            status="ok",
            reasons=_reasons(tag_up[t.track_id][bf], prod_up[p.track_id][bf], cfg),
            candidates=candidates,
        ))

    def _mk_unmatched(trace: TrackTrace, kind: str, best_score: float) -> UnmatchedTrack:
        fr = sorted(trace.boxes)
        return UnmatchedTrack(
            track_id=trace.track_id,
            kind=kind,
            n_frames=trace.n_frames,
            first_frame=fr[0],
            last_frame=fr[-1],
            first_s=fr[0] / fps,
            last_s=fr[-1] / fps,
            representative_bbox=_representative_bbox(trace.boxes),
            persistent=trace.n_frames >= cfg.persistence_min_frames,
            best_score=best_score,
        )

    tag_by_id = {t.track_id: t for t in tags}
    unmatched_tags = tuple(
        _mk_unmatched(tag_by_id[tid], "price_tag", sc)
        for tid, sc in sorted(unmatched_tag_ids)
    )
    unmatched_products = tuple(
        _mk_unmatched(p, "product", best_for_product[p.track_id])
        for p in prods if p.track_id not in matched_product_ids
    )

    return AssociationResult(
        relations=tuple(relations),
        unmatched_price_tags=unmatched_tags,
        unmatched_products=unmatched_products,
    )
