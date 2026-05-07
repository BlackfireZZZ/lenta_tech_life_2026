from __future__ import annotations

import json
import logging
from pathlib import Path

from .aggregator import TrackAggregator
from .config import PipelineConfig
from .detector import YoloTrackerDetector
from .ocr import build_ocr_engine
from .parser import PriceParser
from .rectifier import TagRectifier
from .types import FinalPrediction, PriceObservation

LOGGER = logging.getLogger(__name__)


class PriceTagPipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.detector = YoloTrackerDetector(cfg.detector)
        self.rectifier = TagRectifier(cfg.rectifier)
        self.ocr = build_ocr_engine(cfg.ocr.backend)
        self.parser = PriceParser(cfg.parser)
        self.aggregator = TrackAggregator(cfg.aggregation)

    def run(self, video_path: str, output_jsonl: str | None = None) -> list[FinalPrediction]:
        output_path = output_jsonl or self.cfg.runtime.output_jsonl
        resolved_out = Path(output_path).expanduser().resolve() if output_path else None
        if resolved_out:
            resolved_out.parent.mkdir(parents=True, exist_ok=True)

        finalized: list[FinalPrediction] = []
        for frame_idx, (frame, detections) in enumerate(self.detector.stream_video(video_path)):
            for det in detections:
                if det.track_id is None:
                    continue
                self.aggregator.mark_seen(track_id=det.track_id, frame_idx=det.frame_idx)

                if det.confidence < self.cfg.ocr.min_detection_confidence:
                    continue
                if not self.aggregator.should_run_ocr(
                    track_id=det.track_id,
                    frame_idx=det.frame_idx,
                    min_gap=self.cfg.ocr.min_frames_between_ocr_per_track,
                ):
                    continue

                crop = self.rectifier.rectify(frame, det.bbox_xyxy)
                if crop is None:
                    continue
                if crop.sharpness < self.cfg.ocr.min_sharpness:
                    continue
                if crop.area_px < self.cfg.ocr.min_crop_area_px:
                    continue

                ocr_out = self.ocr.recognize(crop.image)
                if not ocr_out.text.strip():
                    continue
                parsed = self.parser.parse(ocr_out.text, ocr_confidence=ocr_out.confidence)
                if parsed is None:
                    continue

                obs = PriceObservation(
                    frame_idx=det.frame_idx,
                    timestamp_s=det.timestamp_s,
                    track_id=det.track_id,
                    bbox_xyxy=crop.bbox_xyxy,
                    parsed=parsed,
                    detection_confidence=det.confidence,
                    ocr_confidence=ocr_out.confidence,
                    sharpness=crop.sharpness,
                )
                self.aggregator.add_observation(obs)

            expired = self.aggregator.flush_expired(frame_idx)
            if expired:
                finalized.extend(expired)
                if resolved_out:
                    self._append_jsonl(resolved_out, expired)

            if frame_idx % self.cfg.runtime.log_every_n_frames == 0 and frame_idx > 0:
                LOGGER.info(
                    "profile=%s frame=%d detections=%d finalized=%d",
                    self.cfg.runtime.profile_name,
                    frame_idx,
                    len(detections),
                    len(finalized),
                )

        tail = self.aggregator.flush_all()
        if tail:
            finalized.extend(tail)
            if resolved_out:
                self._append_jsonl(resolved_out, tail)
        return finalized

    @staticmethod
    def _append_jsonl(path: Path, rows: list[FinalPrediction]) -> None:
        with path.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

