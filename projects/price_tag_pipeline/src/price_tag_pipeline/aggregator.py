"""Track-level multi-field aggregator with crop buffering and cross-track dedup.

Per-track state holds:
- A buffer of the top-K sharpest crops (for late-OCR-on-best-frames).
- All OCR/VLM observations.

Finalization runs per field independently:
- Numeric fields (prices, weights) are bucketed with a fuzzy tolerance.
- The bucket with the highest summed weight wins.
- A field can be `None` (no signal) and that is its own bucket.

A cross-track deduplication pass runs after all tracks are flushed.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from .config import AggregationConfig
from .quality import tenengrad_sharpness
from .types import (
    CropBufferEntry,
    CropCandidate,
    FinalTag,
    HACK_EXTRA_FIELDS,
    ParsedTag,
    TagObservation,
)


_NUMERIC_EXTRA_FIELDS = {
    "price_discount",
    "discount_amount",
    "price1_qr",
    "price2_qr",
    "price3_qr",
    "price4_qr",
    "wholesale_level_1_count",
    "wholesale_level_1_price",
    "wholesale_level_2_count",
    "wholesale_level_2_price",
    "action_price_qr",
}

# Tenengrad value at which the focus term reaches 0.5 (saturating, scale-free).
# Only the *relative* order of crops matters for buffer ranking, so the exact
# value is non-critical; a reasoned start, sweepable via scripts/eval_tracking.
_FOCUS_HALF = 600.0


def _border_factor(
    bbox: tuple[int, int, int, int], frame_w: int, frame_h: int
) -> float:
    """1.0 for an interior box, decaying to 0.4 as it hugs a frame edge.

    A box touching the frame edge is the tag entering/leaving — motion-blurred
    and truncated, the worst frame to recognise from. Returns 1.0 when frame
    size is unknown (back-compat: no penalty)."""
    if frame_w <= 0 or frame_h <= 0:
        return 1.0
    x1, y1, x2, y2 = bbox
    margin = min(x1, y1, frame_w - x2, frame_h - y2)
    if margin <= 0:
        return 0.4  # truncated at the edge
    band = 0.06 * min(frame_w, frame_h)
    return 0.4 + 0.6 * min(1.0, margin / band) if band > 0 else 1.0


# ---------------------------------------------------------------------------
# Per-track state
# ---------------------------------------------------------------------------

@dataclass
class _TrackState:
    observations: list[TagObservation] = field(default_factory=list)
    crops: list[CropBufferEntry] = field(default_factory=list)
    last_seen_frame: int = -1
    last_ocr_frame: int = -10_000
    last_bbox: tuple[int, int, int, int] = (0, 0, 0, 0)


class TrackAggregator:
    def __init__(self, cfg: AggregationConfig):
        self.cfg = cfg
        self._tracks: dict[int, _TrackState] = {}

    # ----- track maintenance -----

    def mark_seen(self, track_id: int, frame_idx: int, bbox: tuple[int, int, int, int]) -> None:
        state = self._tracks.setdefault(track_id, _TrackState())
        state.last_seen_frame = frame_idx
        state.last_bbox = bbox

    def should_run_ocr(self, track_id: int, frame_idx: int, min_gap: int) -> bool:
        state = self._tracks.get(track_id)
        if state is None:
            return True
        return frame_idx - state.last_ocr_frame >= min_gap

    def push_crop(
        self,
        track_id: int,
        crop: CropCandidate,
        frame_w: int = 0,
        frame_h: int = 0,
    ) -> None:
        """Add a candidate crop to a track's top-K-best buffer.

        Ranking is tracker-layer only (which of a track's frames we keep as
        "best" — strategy §4.2); it does not touch recognition. Tenengrad
        focus dominates (motion-blur-robust, saturating so one very sharp
        frame can't swamp the other terms); a box hugging the frame edge is
        penalised because that is the tag entering/leaving (blurred +
        truncated). ``frame_w/h`` are optional for back-compat.
        """
        state = self._tracks.setdefault(track_id, _TrackState())
        img = crop.image
        focus = (
            tenengrad_sharpness(img)
            if img is not None and img.size
            else float(crop.sharpness)
        )
        sharp_term = focus / (focus + _FOCUS_HALF)  # 0..1, scale-free
        area_term = min(1.0, crop.area_px / 20_000.0)
        quality = (
            0.60 * sharp_term
            + 0.15 * area_term
            + 0.25 * crop.detection_confidence
        )
        quality *= _border_factor(crop.bbox_xyxy, frame_w, frame_h)
        entry = CropBufferEntry(crop=crop, quality_score=quality)
        state.crops.append(entry)
        # Keep only the top-K by quality.
        # Sort once after each insert is fine; K is small.
        # We rely on the pipeline's `min_frames_between_ocr_per_track` to keep
        # the buffer growth bounded.
        if len(state.crops) > 64:
            state.crops.sort(key=lambda e: e.quality_score, reverse=True)
            del state.crops[64:]

    def best_crops(self, track_id: int, k: int) -> list[CropBufferEntry]:
        state = self._tracks.get(track_id)
        if state is None or not state.crops:
            return []
        state.crops.sort(key=lambda e: e.quality_score, reverse=True)
        return state.crops[:k]

    def clear_crops(self, track_id: int) -> None:
        state = self._tracks.get(track_id)
        if state is None:
            return
        state.crops.clear()

    def add_observation(self, obs: TagObservation) -> None:
        state = self._tracks.setdefault(obs.track_id, _TrackState())
        state.observations.append(obs)
        state.last_seen_frame = obs.frame_idx
        state.last_ocr_frame = obs.frame_idx
        state.last_bbox = obs.bbox_xyxy

    # ----- flushing -----

    def flush_expired(self, frame_idx: int) -> list[FinalTag]:
        expired_ids = [
            tid
            for tid, state in self._tracks.items()
            if state.last_seen_frame >= 0
            and frame_idx - state.last_seen_frame > self.cfg.track_ttl_frames
        ]
        out: list[FinalTag] = []
        for tid in expired_ids:
            state = self._tracks.pop(tid)
            final = self._finalize_track(tid, state)
            if final is not None:
                out.append(final)
        return out

    def flush_all(self) -> list[FinalTag]:
        out: list[FinalTag] = []
        for tid, state in list(self._tracks.items()):
            final = self._finalize_track(tid, state)
            if final is not None:
                out.append(final)
        self._tracks.clear()
        return out

    # ----- finalization -----

    def _finalize_track(self, track_id: int, state: _TrackState) -> Optional[FinalTag]:
        obs_list = state.observations
        if len(obs_list) < self.cfg.min_observations_per_track:
            return None

        # Per-field voting.
        regular, regular_conf = self._vote_numeric(obs_list, "regular_price", "regular_price_confidence")
        loyalty, loyalty_conf = self._vote_numeric(obs_list, "loyalty_price", "loyalty_price_confidence")
        weight_value, weight_conf = self._vote_numeric(obs_list, "weight_value", "weight_confidence")
        ppu_value, ppu_conf = self._vote_numeric(obs_list, "price_per_unit_value", "price_per_unit_confidence")
        weight_unit = self._vote_categorical(obs_list, "weight_unit", "weight_confidence")
        ppu_unit = self._vote_categorical(obs_list, "price_per_unit_unit", "price_per_unit_confidence")
        name, name_conf = self._vote_text(obs_list, "product_name", "product_name_confidence")
        promo = self._vote_promo(obs_list)
        currency = self._vote_categorical(obs_list, "currency", None) or "RUB"
        extra_fields: dict[str, object] = {}
        extra_confidences: dict[str, float] = {}
        for key in HACK_EXTRA_FIELDS:
            value, conf = self._vote_extra_field(obs_list, key)
            if value not in (None, ""):
                extra_fields[key] = value
                extra_confidences[key] = conf

        field_confidences = {
            "regular_price": regular_conf,
            "loyalty_price": loyalty_conf,
            "weight": weight_conf,
            "price_per_unit": ppu_conf,
            "product_name": name_conf,
        }
        field_confidences.update({f"extra:{k}": v for k, v in extra_confidences.items()})
        non_zero = [v for v in field_confidences.values() if v > 0]
        overall_conf = (sum(non_zero) / len(non_zero)) if non_zero else 0.0

        if overall_conf < self.cfg.min_final_confidence:
            return None

        # Pick the BEST observation for the reported bbox/timestamp — the
        # frame where recognition was strongest, NOT state.last_bbox (the tag
        # leaving the frame edge: motion-blurred and truncated). The hackathon
        # matches a no-barcode row by frame_timestamp+bbox (briefing §3.4/§5),
        # so emitting the leaving-frame box loses spatio-temporal matches.
        repr_obs = max(obs_list, key=lambda o: o.field_weight(1.0))

        weight_unit_str = weight_unit.value if hasattr(weight_unit, "value") else weight_unit

        return FinalTag(
            track_id=track_id,
            bbox_xyxy=repr_obs.bbox_xyxy,
            timestamp_s=repr_obs.timestamp_s,
            source_frames=sorted({o.frame_idx for o in obs_list}),
            regular_price=regular,
            loyalty_price=loyalty,
            product_name=name,
            weight_value=weight_value,
            weight_unit=weight_unit_str,
            price_per_unit_value=ppu_value,
            price_per_unit_unit=ppu_unit,
            promo_flag=promo,
            currency=str(currency) if currency else "RUB",
            field_confidences=field_confidences,
            extra_fields=extra_fields,
            overall_confidence=overall_conf,
            n_observations=len(obs_list),
        )

    # ----- voting primitives -----

    def _vote_numeric(
        self,
        obs_list: list[TagObservation],
        value_attr: str,
        conf_attr: str,
    ) -> tuple[Optional[float], float]:
        tol = self.cfg.price_fuzzy_tolerance
        buckets: list[tuple[float, float, int]] = []  # (center_value, total_weight, count)
        none_weight = 0.0
        for obs in obs_list:
            v = getattr(obs.parsed, value_attr)
            c = getattr(obs.parsed, conf_attr)
            w = obs.field_weight(c)
            if v is None or c <= 0:
                none_weight += w
                continue
            placed = False
            for i, (center, weight, count) in enumerate(buckets):
                if abs(center - v) <= tol:
                    new_center = (center * count + v) / (count + 1)
                    buckets[i] = (new_center, weight + w, count + 1)
                    placed = True
                    break
            if not placed:
                buckets.append((v, w, 1))

        if not buckets:
            return None, 0.0
        best = max(buckets, key=lambda b: b[1])
        if best[1] < none_weight:
            return None, 0.0
        total = sum(b[1] for b in buckets) + none_weight
        confidence = best[1] / total if total > 0 else 0.0
        return round(best[0], 2), confidence

    def _vote_categorical(
        self,
        obs_list: list[TagObservation],
        value_attr: str,
        conf_attr: Optional[str],
    ):
        votes: defaultdict[object, float] = defaultdict(float)
        for obs in obs_list:
            v = getattr(obs.parsed, value_attr)
            if v is None:
                continue
            c = getattr(obs.parsed, conf_attr) if conf_attr else 1.0
            if c <= 0:
                continue
            votes[v] += obs.field_weight(c)
        if not votes:
            return None
        return max(votes.items(), key=lambda kv: kv[1])[0]

    def _vote_text(
        self,
        obs_list: list[TagObservation],
        value_attr: str,
        conf_attr: str,
    ) -> tuple[Optional[str], float]:
        # Bucket by normalized lowercase + simple ratio.
        buckets: list[tuple[str, float, list[str]]] = []  # (key, total_weight, originals)
        none_weight = 0.0
        for obs in obs_list:
            v = getattr(obs.parsed, value_attr)
            c = getattr(obs.parsed, conf_attr)
            w = obs.field_weight(c)
            if not v or c <= 0:
                none_weight += w
                continue
            key = " ".join(str(v).lower().split())
            placed = False
            for i, (existing_key, weight, originals) in enumerate(buckets):
                if _name_ratio(existing_key, key) >= self.cfg.name_fuzzy_ratio:
                    originals.append(str(v))
                    buckets[i] = (existing_key, weight + w, originals)
                    placed = True
                    break
            if not placed:
                buckets.append((key, w, [str(v)]))

        if not buckets:
            return None, 0.0
        best = max(buckets, key=lambda b: b[1])
        if best[1] < none_weight:
            return None, 0.0
        total = sum(b[1] for b in buckets) + none_weight
        # Representative: longest original string in the winning bucket (more info).
        rep = max(best[2], key=len)
        return rep, (best[1] / total if total > 0 else 0.0)

    def _vote_extra_field(self, obs_list: list[TagObservation], key: str) -> tuple[object, float]:
        """Vote arbitrary hackathon CSV fields carried in ParsedTag.extra_fields."""
        if key in _NUMERIC_EXTRA_FIELDS:
            return self._vote_extra_numeric(obs_list, key)
        return self._vote_extra_text(obs_list, key)

    def _vote_extra_numeric(self, obs_list: list[TagObservation], key: str) -> tuple[object, float]:
        buckets: list[tuple[float, float, int]] = []
        none_weight = 0.0
        for obs in obs_list:
            raw = obs.parsed.extra_fields.get(key)
            conf = float(obs.parsed.extra_confidences.get(key, 0.0))
            w = obs.field_weight(conf)
            value = _coerce_float(raw)
            if value is None or conf <= 0:
                none_weight += w
                continue
            placed = False
            tol = self.cfg.price_fuzzy_tolerance if "price" in key or "amount" in key else 0.01
            for i, (center, weight, count) in enumerate(buckets):
                if abs(center - value) <= tol:
                    buckets[i] = ((center * count + value) / (count + 1), weight + w, count + 1)
                    placed = True
                    break
            if not placed:
                buckets.append((value, w, 1))
        if not buckets:
            return None, 0.0
        best = max(buckets, key=lambda b: b[1])
        if best[1] < none_weight:
            return None, 0.0
        total = sum(b[1] for b in buckets) + none_weight
        return round(best[0], 2), (best[1] / total if total > 0 else 0.0)

    def _vote_extra_text(self, obs_list: list[TagObservation], key: str) -> tuple[object, float]:
        votes: defaultdict[str, float] = defaultdict(float)
        originals: dict[str, str] = {}
        none_weight = 0.0
        for obs in obs_list:
            raw = obs.parsed.extra_fields.get(key)
            conf = float(obs.parsed.extra_confidences.get(key, 0.0))
            w = obs.field_weight(conf)
            if raw in (None, "") or conf <= 0:
                none_weight += w
                continue
            original = str(raw).strip()
            norm = " ".join(original.lower().split())
            votes[norm] += w
            originals.setdefault(norm, original)
        if not votes:
            return None, 0.0
        best_key, best_weight = max(votes.items(), key=lambda kv: kv[1])
        if best_weight < none_weight:
            return None, 0.0
        total = sum(votes.values()) + none_weight
        return originals[best_key], (best_weight / total if total > 0 else 0.0)

    def _vote_promo(self, obs_list: list[TagObservation]) -> bool:
        on_w = 0.0
        off_w = 0.0
        for obs in obs_list:
            base_conf = max(
                obs.parsed.regular_price_confidence,
                obs.parsed.loyalty_price_confidence,
                0.5,
            )
            w = obs.field_weight(base_conf)
            if obs.parsed.promo_flag:
                on_w += w
            else:
                off_w += w
        return on_w >= off_w * 1.2  # require a 20% margin to call promo


# ---------------------------------------------------------------------------
# Cross-track dedup
# ---------------------------------------------------------------------------

_PRICE_TOL = 0.5


def _tag_barcode(t: FinalTag) -> Optional[str]:
    """The tag's decoded barcode as a digit string, or None.

    Reads fields the recognition layer already produced — this is dedup
    *identity*, not barcode reading (that lives on a separate branch). EAN-8
    is the shortest real GTIN, so shorter digit runs are noise, not a key.
    """
    for key in ("barcode", "qr_code_barcode"):
        v = t.extra_fields.get(key)
        if v in (None, "", "нет"):
            continue
        digits = re.sub(r"\D", "", str(v))
        if len(digits) >= 8:
            return digits
    return None


def _price_relation(a: FinalTag, b: FinalTag) -> str:
    """'agree' | 'conflict' | 'unknown' over co-present price fields."""
    seen = False
    for attr in ("regular_price", "loyalty_price"):
        pa, pb = getattr(a, attr), getattr(b, attr)
        if pa is None or pb is None:
            continue
        seen = True
        if abs(pa - pb) > _PRICE_TOL:
            return "conflict"
    return "agree" if seen else "unknown"


def _name_relation(a: FinalTag, b: FinalTag) -> str:
    if not a.product_name or not b.product_name:
        return "unknown"
    r = _name_ratio(a.product_name.lower(), b.product_name.lower())
    if r >= 0.85:
        return "agree"
    if r < 0.55:
        return "conflict"
    return "weak"


def _content_proximity_merge(
    a: FinalTag, b: FinalTag, iou_threshold: float, time_window_s: float
) -> bool:
    """Merge decision when barcode identity is NOT conclusive.

    Barcode-equal pairs are merged before this is called; barcode-different
    pairs are refused here. This path handles the no-/one-barcode case: it
    needs content agreement plus either box overlap OR — for the moving-camera
    ID-switch where the two fragments sit at different pixels (low IoU) —
    strong corroborating content (price AND name agree).
    """
    ba, bb = _tag_barcode(a), _tag_barcode(b)
    if ba and bb and ba != bb:
        return False  # different physical tags — never merge

    if abs(a.timestamp_s - b.timestamp_s) > time_window_s:
        return False

    price = _price_relation(a, b)
    name = _name_relation(a, b)
    if price == "conflict" or name == "conflict":
        return False
    if price != "agree" and name != "agree":
        return False  # no strong content signal

    if _iou(a.bbox_xyxy, b.bbox_xyxy) >= iou_threshold:
        return True
    # Low IoU: the camera moved between the two fragments. Require BOTH price
    # and name to agree so fragmentation is recovered without merging
    # unrelated tags that merely share a price.
    return price == "agree" and name == "agree"


def dedup_final_tags(
    tags: list[FinalTag],
    iou_threshold: float,
    time_window_s: float,
) -> list[FinalTag]:
    """Collapse predictions describing the same physical tag.

    Identity is content-first, not box-first: the moving robot means two
    fragments of one tag can have near-zero IoU, so an IoU gate would leave
    the duplicate uncollapsed (the metric's #1 enemy, briefing §3.5).
    Strength order:

    1. **Equal decoded barcode ⇒ same tag**, regardless of bbox/time (it is
       the hackathon's primary GT-matching key, briefing §5).
    2. **Different decoded barcodes ⇒ never merged.**
    3. Otherwise: price/name agreement within ``time_window_s`` seconds, with
       IoU a positive — not mandatory — signal.

    ``time_window_s`` is wall-clock seconds (FinalTag.timestamp_s), not
    frames: container FPS varies and frame_timestamp is milliseconds, so a
    frame-count window would mean different durations per video.
    """
    if not tags:
        return []

    parents = list(range(len(tags)))

    def find(x: int) -> int:
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parents[rb] = ra

    # Pass 1 — barcode identity, any time gap.
    by_barcode: defaultdict[str, list[int]] = defaultdict(list)
    for idx, t in enumerate(tags):
        bc = _tag_barcode(t)
        if bc:
            by_barcode[bc].append(idx)
    for idxs in by_barcode.values():
        for k in idxs[1:]:
            union(idxs[0], k)

    # Pass 2 — content + proximity for the rest, time-sorted sliding window.
    order = sorted(range(len(tags)), key=lambda i: tags[i].timestamp_s)
    for ai in range(len(order)):
        i = order[ai]
        for bi in range(ai + 1, len(order)):
            j = order[bi]
            if tags[j].timestamp_s - tags[i].timestamp_s > time_window_s:
                break
            if find(i) == find(j):
                continue
            if _content_proximity_merge(tags[i], tags[j], iou_threshold, time_window_s):
                union(i, j)

    groups: defaultdict[int, list[int]] = defaultdict(list)
    for i in range(len(tags)):
        groups[find(i)].append(i)

    merged: list[FinalTag] = []
    for ids in groups.values():
        if len(ids) == 1:
            merged.append(tags[ids[0]])
            continue
        members = [tags[i] for i in ids]
        # Representative: prefer one carrying a barcode (matches GT by the
        # primary key), then highest confidence.
        rep = max(
            members,
            key=lambda t: (_tag_barcode(t) is not None, t.overall_confidence),
        )
        rep.source_frames = sorted({f for m in members for f in m.source_frames})
        rep.n_observations = sum(m.n_observations for m in members)
        merged.append(rep)
    return merged


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _name_ratio(a: str, b: str) -> float:
    """Cheap similarity score in [0, 1]."""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    # Levenshtein-like ratio using SequenceMatcher (stdlib, no dependency).
    from difflib import SequenceMatcher
    return SequenceMatcher(a=a, b=b).ratio()


def _coerce_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(str(value).replace(",", ".").replace(" ", "").replace("\xa0", ""))
    except (TypeError, ValueError):
        return None
