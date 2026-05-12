"""End-to-end inference pipeline.

Per-frame loop:
    detect -> mark_seen on aggregator -> rectify crop -> push to track buffer
    (no immediate OCR — we OCR the best crops per track at finalization)

On track expiry / video end:
    pick top-K-sharpest crops per track -> OCR/VLM each -> parse -> add observations
    -> aggregator runs per-field voting -> emits FinalTag

Cross-track deduplication runs once at the end on the full list of FinalTags.

This is a meaningful change from the scaffold, which OCR'd every Nth frame
without buffering. Buffering lets us spend OCR/VLM compute on the *sharpest*
crops only, materially improving accuracy at the same call budget.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from typing import Optional

from .aggregator import TrackAggregator, dedup_final_tags
from .config import PipelineConfig
from .detector import build_detector
from .ocr import BaseOCREngine, build_ocr_engine
from .parser import TagParser
from .rectifier import build_rectifier
from .types import CropCandidate, FinalTag, ParsedTag, TagObservation

LOGGER = logging.getLogger(__name__)


class PriceTagPipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.detector = build_detector(cfg.detector)
        self.rectifier = build_rectifier(cfg.rectifier)
        self.ocr: BaseOCREngine = build_ocr_engine(cfg.ocr)
        self.parser = TagParser(cfg.parser)
        self.aggregator = TrackAggregator(cfg.aggregation)
        self._sr = None
        if cfg.rectifier.super_resolution:
            from .super_resolution import SuperResolution
            self._sr = SuperResolution(scale=2)
        self._audit_path: Optional[Path] = None
        if cfg.runtime.audit_path:
            self._audit_path = Path(cfg.runtime.audit_path).expanduser().resolve()
            self._audit_path.parent.mkdir(parents=True, exist_ok=True)
            self._audit_path.write_text("", encoding="utf-8")  # truncate on start

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def run(self, video_path: str, output_path: Optional[str] = None) -> list[FinalTag]:
        resolved_out = self._resolve_output(output_path)
        finalized: list[FinalTag] = []

        for frame_idx, (frame, detections) in enumerate(
            self.detector.stream_video(video_path, fps_override=self.cfg.runtime.fps_override)
        ):
            for det in detections:
                if det.track_id is None:
                    continue
                self.aggregator.mark_seen(det.track_id, det.frame_idx, det.bbox_xyxy)

                if det.confidence < self.cfg.ocr.min_detection_confidence:
                    continue

                crop = self.rectifier.rectify(frame, det)
                if crop is None:
                    continue
                if crop.sharpness < self.cfg.ocr.min_sharpness:
                    continue
                if crop.area_px < self.cfg.ocr.min_crop_area_px:
                    continue

                crop = self._maybe_upscale(crop)
                self.aggregator.push_crop(det.track_id, crop)

            # Tracks whose last_seen is older than TTL get finalized now.
            expiring = self._find_expiring_track_ids(frame_idx)
            for tid in expiring:
                self._run_ocr_for_track(tid)

            new_finals = self.aggregator.flush_expired(frame_idx)
            if new_finals:
                finalized.extend(new_finals)

            if (
                frame_idx > 0
                and self.cfg.runtime.log_every_n_frames > 0
                and frame_idx % self.cfg.runtime.log_every_n_frames == 0
            ):
                LOGGER.info(
                    "profile=%s frame=%d detections=%d finalized=%d",
                    self.cfg.runtime.profile_name,
                    frame_idx,
                    len(detections),
                    len(finalized),
                )

        # Flush any tracks still live at end of video.
        for tid in list(self.aggregator._tracks.keys()):
            self._run_ocr_for_track(tid)
        finalized.extend(self.aggregator.flush_all())

        # Cross-track deduplication on the full list.
        deduped = dedup_final_tags(
            finalized,
            iou_threshold=self.cfg.aggregation.dedup_iou_threshold,
            time_window_frames=self.cfg.aggregation.dedup_time_window_frames,
        )
        LOGGER.info(
            "Finalized %d -> %d after dedup (saved %d duplicates).",
            len(finalized), len(deduped), len(finalized) - len(deduped),
        )

        if resolved_out:
            self._write_output(resolved_out, deduped)

        return deduped

    # -----------------------------------------------------------------
    # Internal: OCR-on-best-crops
    # -----------------------------------------------------------------

    def _find_expiring_track_ids(self, frame_idx: int) -> list[int]:
        ttl = self.cfg.aggregation.track_ttl_frames
        return [
            tid
            for tid, state in self.aggregator._tracks.items()
            if state.last_seen_frame >= 0
            and frame_idx - state.last_seen_frame > ttl
        ]

    def _run_ocr_for_track(self, track_id: int) -> None:
        """OCR the top-K-sharpest crops in the buffer and turn them into observations.

        Supports ensemble engines: one crop may produce multiple OCRResults,
        each becoming a separate observation that the aggregator votes on.
        """
        k = self.cfg.ocr.top_k_crops_per_track
        crops = self.aggregator.best_crops(track_id, k)
        if not crops:
            return

        for entry in crops:
            try:
                results = self.ocr.recognize_all(entry.crop.image)
            except Exception as exc:  # do not abort the whole video on a single OCR fail
                LOGGER.warning("OCR failed on track=%d frame=%d: %s", track_id, entry.crop.frame_idx, exc)
                continue
            for res in results:
                if not res.text:
                    continue
                if self.ocr.structured or res.text.lstrip().startswith("{"):
                    parsed = self.parser.parse_vlm_json(
                        res.text, vlm_confidence=res.confidence, backend=res.backend
                    )
                else:
                    parsed = self.parser.parse_text(
                        res.text, ocr_confidence=res.confidence, backend=res.backend
                    )
                self._audit(track_id, entry.crop.frame_idx, res, parsed)
                if self._parsed_is_empty(parsed):
                    continue
                obs = TagObservation(
                    frame_idx=entry.crop.frame_idx,
                    timestamp_s=entry.crop.timestamp_s,
                    track_id=track_id,
                    bbox_xyxy=entry.crop.bbox_xyxy,
                    parsed=parsed,
                    detection_confidence=entry.crop.detection_confidence,
                    sharpness=entry.crop.sharpness,
                )
                self.aggregator.add_observation(obs)

        # Drop the buffer once we have committed observations.
        self.aggregator.clear_crops(track_id)

    def _audit(self, track_id: int, frame_idx: int, res, parsed: ParsedTag) -> None:
        """Append one JSONL line describing this OCR call. No-op when disabled."""
        if self._audit_path is None:
            return
        try:
            row = {
                "track_id": int(track_id),
                "frame_idx": int(frame_idx),
                "backend": res.backend,
                "ocr_confidence": float(res.confidence),
                "raw_text": res.text,
                "parsed": {
                    "regular_price": parsed.regular_price,
                    "loyalty_price": parsed.loyalty_price,
                    "product_name": parsed.product_name,
                    "weight_value": parsed.weight_value,
                    "weight_unit": (parsed.weight_unit.value
                                    if parsed.weight_unit is not None else None),
                    "price_per_unit_value": parsed.price_per_unit_value,
                    "price_per_unit_unit": parsed.price_per_unit_unit,
                    "promo_flag": parsed.promo_flag,
                    "currency": parsed.currency,
                },
            }
            with self._audit_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception as exc:
            LOGGER.warning("Audit write failed: %s", exc)

    def _maybe_upscale(self, crop: CropCandidate) -> CropCandidate:
        if self._sr is None or crop.area_px >= self.cfg.rectifier.sr_min_area_px:
            return crop
        try:
            upscaled = self._sr.upscale(crop.image)
        except Exception as exc:  # never let SR kill the pipeline
            LOGGER.warning("Super-resolution failed for frame=%d: %s", crop.frame_idx, exc)
            return crop
        new_area = int(upscaled.shape[0] * upscaled.shape[1])
        return CropCandidate(
            image=upscaled,
            sharpness=crop.sharpness,
            area_px=new_area,
            bbox_xyxy=crop.bbox_xyxy,
            detection_confidence=crop.detection_confidence,
            frame_idx=crop.frame_idx,
            timestamp_s=crop.timestamp_s,
        )

    @staticmethod
    def _parsed_is_empty(parsed: ParsedTag) -> bool:
        return (
            parsed.regular_price is None
            and parsed.loyalty_price is None
            and parsed.product_name is None
            and parsed.weight_value is None
            and parsed.price_per_unit_value is None
        )

    # -----------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------

    def _resolve_output(self, override: Optional[str]) -> Optional[Path]:
        path = override or self.cfg.runtime.output_path
        if not path:
            return None
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        # Truncate so re-runs do not accumulate stale predictions.
        p.write_text("", encoding="utf-8")
        return p

    @staticmethod
    def _write_output(path: Path, rows: list[FinalTag]) -> None:
        # JSONL preferred. If the caller asked for .json we wrap as array; otherwise jsonl.
        if path.suffix.lower() == ".json":
            with path.open("w", encoding="utf-8") as f:
                json.dump([r.to_dict() for r in rows], f, ensure_ascii=False, indent=2)
        else:
            with path.open("w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
