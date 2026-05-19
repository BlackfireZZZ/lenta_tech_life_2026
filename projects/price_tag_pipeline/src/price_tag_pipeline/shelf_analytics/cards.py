"""Group sibling product-facing tracks into one product → ProductCard set.

A price tag serves a whole SKU group (a row of identical packs), but the
detector+tracker yields one track per *facing* (per bottle). Alerting per
facing massively over-counts MISSING_PRICE_TAG. This module collapses adjacent,
contemporaneous, similar-looking facing tracks into one product group so alerts
and the UI 'cards' are product-level.

Pure-Python and deterministic (no cv2 / no OCR): visual similarity is *injected*
as an opaque appearance vector + similarity fn, so the runner owns colour-hist
extraction and this stays unit-testable and off the graded CSV path. Price /
name / barcode on the card are left None here and filled at integration time
from the real recognition pipeline (see docs/shelf-audit.md §8).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from math import sqrt

from .associate import AssociationResult, TrackTrace
from .geometry import BBox, box_height, box_width, center, to_upright
from .schema import ProductCard


@dataclass(frozen=True)
class CardConfig:
    rotation: str = "ccw"
    band_overlap_min: float = 0.30   # vertical (shelf-row) overlap of rep boxes
    max_gap_ratio: float = 1.8       # horizontal gap ≤ N·avg_width = adjacent
    max_frame_gap: int = 12          # sibling tracks seen within N frames
    min_appearance_sim: float = 0.75  # only applied when appearance is provided


@dataclass(frozen=True)
class CardSet:
    cards: tuple[ProductCard, ...]
    track_to_card: dict[int, str]
    card_members: dict[str, tuple[int, ...]]

    def to_dict(self) -> dict:
        return {
            "cards": [c.to_dict() for c in self.cards],
            "track_to_card": {str(k): v for k, v in self.track_to_card.items()},
            "card_members": {k: list(v) for k, v in self.card_members.items()},
        }


def _cosine(a, b) -> float:
    if a is None or b is None or len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = sqrt(sum(x * x for x in a))
    nb = sqrt(sum(y * y for y in b))
    return 0.0 if na <= 0 or nb <= 0 else dot / (na * nb)


def _rep_upright(trace: TrackTrace, fw: int, fh: int, rot: str) -> BBox:
    ups = [to_upright(b, fw, fh, rot) for b in trace.boxes.values()]
    xs1 = sorted(u[0] for u in ups)
    ys1 = sorted(u[1] for u in ups)
    xs2 = sorted(u[2] for u in ups)
    ys2 = sorted(u[3] for u in ups)
    m = len(ups) // 2
    return (xs1[m], ys1[m], xs2[m], ys2[m])


def _band_overlap(a: BBox, b: BBox) -> float:
    inter = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    denom = max(1.0, min(box_height(a), box_height(b)))
    return inter / denom


def _x_gap(a: BBox, b: BBox) -> float:
    if a[2] <= b[0]:
        return b[0] - a[2]
    if b[2] <= a[0]:
        return a[0] - b[2]
    return 0.0  # overlapping in x


def _frame_gap(a: TrackTrace, b: TrackTrace) -> int:
    fa, fb = sorted(a.boxes), sorted(b.boxes)
    if fa[0] <= fb[-1] and fb[0] <= fa[-1]:
        return 0  # frame ranges overlap
    return min(abs(fa[0] - fb[-1]), abs(fb[0] - fa[-1]))


def _groupable(a: TrackTrace, b: TrackTrace, ra: BBox, rb: BBox,
                appearance, sim_fn, cfg: CardConfig) -> bool:
    if _band_overlap(ra, rb) < cfg.band_overlap_min:
        return False
    avg_w = max(1.0, (box_width(ra) + box_width(rb)) / 2.0)
    if _x_gap(ra, rb) > cfg.max_gap_ratio * avg_w:
        return False
    if _frame_gap(a, b) > cfg.max_frame_gap:
        return False
    if appearance is not None:
        va, vb = appearance.get(a.track_id), appearance.get(b.track_id)
        if va is not None and vb is not None:
            if sim_fn(va, vb) < cfg.min_appearance_sim:
                return False
    return True


def build_card_set(
    product_traces: list[TrackTrace] | tuple[TrackTrace, ...],
    frame_w: int,
    frame_h: int,
    fps: float,
    video_id: str,
    appearance: dict[int, tuple] | None = None,
    sim_fn: Callable | None = None,
    best_track: Callable[[int], float] | None = None,
    cfg: CardConfig | None = None,
) -> CardSet:
    """Cluster facing tracks into product groups → one ProductCard per group.

    ``appearance`` maps track_id → opaque vector; ``sim_fn`` scores two vectors
    in [0,1] (default cosine). ``best_track`` maps track_id → a quality score
    (e.g. best-crop sharpness) used to pick the card's representative track;
    falls back to longest track. Deterministic: components sorted by smallest
    member track_id.
    """

    cfg = cfg or CardConfig()
    sim_fn = sim_fn or _cosine
    fps = fps if fps and fps > 1e-6 else 1.0
    traces = sorted(product_traces, key=lambda t: t.track_id)
    rep = {t.track_id: _rep_upright(t, frame_w, frame_h, cfg.rotation) for t in traces}

    # Union-find over groupable pairs.
    parent = {t.track_id: t.track_id for t in traces}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)

    for i, a in enumerate(traces):
        for b in traces[i + 1:]:
            if _groupable(a, b, rep[a.track_id], rep[b.track_id],
                          appearance, sim_fn, cfg):
                union(a.track_id, b.track_id)

    comps: dict[int, list[TrackTrace]] = {}
    for t in traces:
        comps.setdefault(find(t.track_id), []).append(t)

    by_id = {t.track_id: t for t in traces}
    cards: list[ProductCard] = []
    track_to_card: dict[int, str] = {}
    card_members: dict[str, tuple[int, ...]] = {}

    for idx, root in enumerate(sorted(comps), start=1):
        members = sorted(comps[root], key=lambda t: t.track_id)
        mids = tuple(m.track_id for m in members)
        if best_track is not None:
            rep_tid = max(mids, key=lambda tid: (best_track(tid), by_id[tid].n_frames))
        else:
            rep_tid = max(mids, key=lambda tid: by_id[tid].n_frames)
        all_frames = sorted(f for m in members for f in m.boxes)
        card_id = f"card_{video_id}_{idx:04d}"
        cards.append(ProductCard(
            card_id=card_id,
            product_track_id=rep_tid,
            facing_count=len(mids),
            seen_from_s=all_frames[0] / fps,
            seen_to_s=all_frames[-1] / fps,
        ))
        card_members[card_id] = mids
        for tid in mids:
            track_to_card[tid] = card_id

    return CardSet(tuple(cards), track_to_card, card_members)


def regroup_missing_price_tag(
    assoc: AssociationResult,
    card_set: CardSet,
) -> list[str]:
    """Card ids whose every member facing is unmatched → MISSING_PRICE_TAG.

    Collapses facing-level over-alerting: a group is tag-covered if ANY member
    is the product of an ``ok`` relation.
    """
    covered_tracks: set[int] = set()
    for r in assoc.relations:
        if r.status == "ok" and r.product_group_id:
            covered_tracks.add(int(r.product_group_id.split("_")[-1]))
    covered_cards = {card_set.track_to_card[t] for t in covered_tracks
                     if t in card_set.track_to_card}
    return [c.card_id for c in card_set.cards if c.card_id not in covered_cards]
