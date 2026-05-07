from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from .config import AggregationConfig
from .types import FinalPrediction, PriceObservation


@dataclass
class _TrackState:
    observations: list[PriceObservation] = field(default_factory=list)
    last_seen_frame: int = -1
    last_ocr_frame: int = -10_000


class TrackAggregator:
    def __init__(self, cfg: AggregationConfig):
        self.cfg = cfg
        self._tracks: dict[int, _TrackState] = {}

    def should_run_ocr(self, track_id: int, frame_idx: int, min_gap: int) -> bool:
        state = self._tracks.get(track_id)
        if state is None:
            return True
        return frame_idx - state.last_ocr_frame >= min_gap

    def add_observation(self, obs: PriceObservation) -> None:
        state = self._tracks.setdefault(obs.track_id, _TrackState())
        state.observations.append(obs)
        state.last_seen_frame = obs.frame_idx
        state.last_ocr_frame = obs.frame_idx

    def mark_seen(self, track_id: int, frame_idx: int) -> None:
        state = self._tracks.setdefault(track_id, _TrackState())
        state.last_seen_frame = frame_idx

    def flush_expired(self, frame_idx: int) -> list[FinalPrediction]:
        expired_ids = [
            track_id
            for track_id, state in self._tracks.items()
            if state.last_seen_frame >= 0 and frame_idx - state.last_seen_frame > self.cfg.track_ttl_frames
        ]
        preds: list[FinalPrediction] = []
        for track_id in expired_ids:
            state = self._tracks.pop(track_id)
            pred = self._finalize_track(track_id, state.observations)
            if pred is not None:
                preds.append(pred)
        return preds

    def flush_all(self) -> list[FinalPrediction]:
        preds: list[FinalPrediction] = []
        for track_id, state in list(self._tracks.items()):
            pred = self._finalize_track(track_id, state.observations)
            if pred is not None:
                preds.append(pred)
        self._tracks.clear()
        return preds

    def _finalize_track(self, track_id: int, obs_list: list[PriceObservation]) -> Optional[FinalPrediction]:
        if len(obs_list) < self.cfg.min_observations_per_track:
            return None

        votes = defaultdict(float)
        by_price = defaultdict(list)
        for obs in obs_list:
            votes[obs.parsed.normalized_price] += obs.score
            by_price[obs.parsed.normalized_price].append(obs)

        best_price = max(votes.keys(), key=lambda k: votes[k])
        bucket = by_price[best_price]
        total_score = sum(votes.values())
        best_score = votes[best_price]
        final_conf = best_score / total_score if total_score > 0 else 0.0
        if final_conf < self.cfg.min_final_confidence:
            return None

        repr_obs = max(bucket, key=lambda x: x.score)
        return FinalPrediction(
            price=repr_obs.parsed.normalized_price,
            price_value=repr_obs.parsed.value,
            currency=repr_obs.parsed.currency,
            unit=repr_obs.parsed.unit,
            confidence=final_conf,
            track_id=track_id,
            bbox_xyxy=repr_obs.bbox_xyxy,
            timestamp_s=repr_obs.timestamp_s,
            source_frames=sorted({o.frame_idx for o in bucket}),
        )

