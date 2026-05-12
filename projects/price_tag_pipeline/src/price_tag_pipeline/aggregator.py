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

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from .config import AggregationConfig
from .types import (
    CropBufferEntry,
    CropCandidate,
    FinalTag,
    ParsedTag,
    TagObservation,
)


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

    def push_crop(self, track_id: int, crop: CropCandidate) -> None:
        """Add a candidate crop to a track's top-K-sharpest buffer."""
        state = self._tracks.setdefault(track_id, _TrackState())
        # Composite quality: sharpness dominates, area and det_conf adjust.
        quality = (
            0.6 * min(1.0, crop.sharpness / 200.0)
            + 0.2 * min(1.0, crop.area_px / 20_000.0)
            + 0.2 * crop.detection_confidence
        )
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

        field_confidences = {
            "regular_price": regular_conf,
            "loyalty_price": loyalty_conf,
            "weight": weight_conf,
            "price_per_unit": ppu_conf,
            "product_name": name_conf,
        }
        non_zero = [v for v in field_confidences.values() if v > 0]
        overall_conf = (sum(non_zero) / len(non_zero)) if non_zero else 0.0

        if overall_conf < self.cfg.min_final_confidence:
            return None

        # Pick a representative observation for bbox/timestamp.
        repr_obs = max(obs_list, key=lambda o: o.field_weight(1.0))

        weight_unit_str = weight_unit.value if hasattr(weight_unit, "value") else weight_unit

        return FinalTag(
            track_id=track_id,
            bbox_xyxy=state.last_bbox or repr_obs.bbox_xyxy,
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

def dedup_final_tags(
    tags: list[FinalTag],
    iou_threshold: float,
    time_window_frames: int,
) -> list[FinalTag]:
    """Merge predictions that almost certainly describe the same physical tag.

    Two tags are merged when their bboxes overlap above `iou_threshold`,
    they fall inside `time_window_frames` of each other, and at least one
    of (regular_price, loyalty_price, product_name) matches.
    """
    if not tags:
        return []
    # Sort by start frame.
    tags = sorted(tags, key=lambda t: t.source_frames[0] if t.source_frames else 0)
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

    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            ti, tj = tags[i], tags[j]
            # Time-window guard.
            si = ti.source_frames[-1] if ti.source_frames else 0
            sj = tj.source_frames[0] if tj.source_frames else 0
            if sj - si > time_window_frames:
                break
            if _iou(ti.bbox_xyxy, tj.bbox_xyxy) < iou_threshold:
                continue
            if not _content_matches(ti, tj):
                continue
            union(i, j)

    groups: defaultdict[int, list[int]] = defaultdict(list)
    for i in range(len(tags)):
        groups[find(i)].append(i)

    merged: list[FinalTag] = []
    for ids in groups.values():
        if len(ids) == 1:
            merged.append(tags[ids[0]])
            continue
        # Pick the highest-confidence as representative; merge source_frames.
        members = [tags[i] for i in ids]
        rep = max(members, key=lambda t: t.overall_confidence)
        merged_frames = sorted({f for m in members for f in m.source_frames})
        rep.source_frames = merged_frames
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


def _content_matches(a: FinalTag, b: FinalTag) -> bool:
    # Prices are the strongest signal — if both have regular prices that disagree, refuse merge.
    if a.regular_price is not None and b.regular_price is not None:
        if abs(a.regular_price - b.regular_price) > 0.5:
            return False
        return True
    if a.loyalty_price is not None and b.loyalty_price is not None:
        if abs(a.loyalty_price - b.loyalty_price) > 0.5:
            return False
        return True
    if a.product_name and b.product_name:
        return _name_ratio(a.product_name.lower(), b.product_name.lower()) >= 0.80
    return False


def _name_ratio(a: str, b: str) -> float:
    """Cheap similarity score in [0, 1]."""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    # Levenshtein-like ratio using SequenceMatcher (stdlib, no dependency).
    from difflib import SequenceMatcher
    return SequenceMatcher(a=a, b=b).ratio()
