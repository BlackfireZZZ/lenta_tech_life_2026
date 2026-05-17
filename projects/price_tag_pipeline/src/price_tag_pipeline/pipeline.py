"""End-to-end inference pipeline.

Per-frame loop:
    detect -> mark_seen on aggregator -> rectify crop -> push to track buffer
    (no immediate OCR — we OCR the best crops per track at finalization)

On track expiry / video end:
    pick top-K-sharpest crops per track -> run the recognition chain on each
    (QR -> barcode -> smart OCR) -> add every reading as an observation
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

from .aggregator import TrackAggregator, dedup_final_tags
from .config import PipelineConfig
from .detector import build_detector, read_video_frame_count
from .progress import Phase, ProgressEvent, ProgressLike, ProgressReporter, as_reporter
from .recognition import RecognitionChain, build_recognition_chain, parsed_is_empty
from .rectifier import build_rectifier
from .types import CropCandidate, FinalTag, TagObservation

LOGGER = logging.getLogger(__name__)

# Progress fraction budget per phase. Detection (the per-frame loop, which
# also runs OCR on expiring tracks inline) is by far the bulk of the
# wall-clock time, so it owns almost the whole bar; the end-of-video
# finalize + cross-track dedup tail is at most a few seconds.
_DETECT_CEIL = 0.97
_FINALIZE_FRAC = 0.97
_DEDUP_FRAC = 0.99


class PriceTagPipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.detector = build_detector(cfg.detector)
        self.rectifier = build_rectifier(cfg.rectifier)
        # QR -> barcode -> smart OCR, behind one stable seam. Parser/OCR/QR
        # wiring now lives in price_tag_pipeline.recognition, not here.
        self.recognition: RecognitionChain = build_recognition_chain(cfg)
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

    def run(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        progress: ProgressLike = None,
    ) -> list[FinalTag]:
        """Run the full pipeline on one video.

        ``progress`` is optional and backwards compatible: ``None`` (default)
        is a silent no-op, or pass a :class:`~price_tag_pipeline.progress.
        ProgressReporter` or a bare ``callable(ProgressEvent)``. It is the
        single signal the CLI bar, the Gradio UI and the ML-service poll
        endpoint all consume. See docs/pipeline-reference.md "Progress".
        """
        reporter = as_reporter(progress)
        resolved_out = self._resolve_output(output_path)
        finalized: list[FinalTag] = []

        frames_total = read_video_frame_count(video_path)
        # Report at most once every `report_step` frames to keep the bar
        # cheap over a websocket; phase changes always report.
        report_step = self.cfg.runtime.log_every_n_frames
        if report_step <= 0:
            report_step = 30
        reporter.publish(ProgressEvent(
            phase=Phase.DETECT, fraction=0.0,
            frames_done=0, frames_total=frames_total,
            message="starting",
        ))

        try:
            self._run(
                video_path, frames_total, report_step, finalized, reporter,
            )
            deduped = self._finalize(finalized, frames_total, reporter)
            if resolved_out:
                self._write_output(resolved_out, deduped)
            reporter.publish(ProgressEvent(
                phase=Phase.DONE, fraction=1.0,
                frames_done=frames_total, frames_total=frames_total,
                tags_finalized=len(deduped), message="complete",
            ))
            return deduped
        finally:
            reporter.close()

    def _run(
        self,
        video_path: str,
        frames_total: int,
        report_step: int,
        finalized: list[FinalTag],
        reporter: ProgressReporter,
    ) -> None:
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
                self._recognize_track(tid)

            new_finals = self.aggregator.flush_expired(frame_idx)
            if new_finals:
                finalized.extend(new_finals)

            if frame_idx > 0 and frame_idx % report_step == 0:
                LOGGER.info(
                    "profile=%s frame=%d detections=%d finalized=%d",
                    self.cfg.runtime.profile_name,
                    frame_idx,
                    len(detections),
                    len(finalized),
                )
                done = frame_idx + 1
                frac = (done / frames_total) * _DETECT_CEIL if frames_total else 0.0
                reporter.publish(ProgressEvent(
                    phase=Phase.DETECT, fraction=frac,
                    frames_done=done, frames_total=frames_total,
                    tags_finalized=len(finalized),
                ))

    def _finalize(
        self,
        finalized: list[FinalTag],
        frames_total: int,
        reporter: ProgressReporter,
    ) -> list[FinalTag]:
        # Flush any tracks still live at end of video (OCR their best crops).
        reporter.publish(ProgressEvent(
            phase=Phase.FINALIZE, fraction=_FINALIZE_FRAC,
            frames_done=frames_total, frames_total=frames_total,
            tags_finalized=len(finalized), message="finalizing tracks",
        ))
        for tid in list(self.aggregator._tracks.keys()):
            self._recognize_track(tid)
        finalized.extend(self.aggregator.flush_all())

        # Cross-track deduplication on the full list.
        reporter.publish(ProgressEvent(
            phase=Phase.DEDUP, fraction=_DEDUP_FRAC,
            frames_done=frames_total, frames_total=frames_total,
            tags_finalized=len(finalized), message="deduplicating",
        ))
        deduped = dedup_final_tags(
            finalized,
            iou_threshold=self.cfg.aggregation.dedup_iou_threshold,
            time_window_s=self.cfg.aggregation.dedup_time_window_s,
        )
        LOGGER.info(
            "Finalized %d -> %d after dedup (saved %d duplicates).",
            len(finalized), len(deduped), len(finalized) - len(deduped),
        )
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

    def _recognize_track(self, track_id: int) -> None:
        """Run the recognition chain on the top-K-sharpest crops of a track.

        Each crop goes through ``QR -> barcode -> smart OCR``; every non-empty
        reading becomes one observation the aggregator votes on (an ensemble
        OCR engine yields several per crop). Per-field reconciliation between
        QR/barcode/OCR is the aggregator's weighted voting, not done here.
        """
        k = self.cfg.ocr.top_k_crops_per_track
        crops = self.aggregator.best_crops(track_id, k)
        if not crops:
            return

        for entry in crops:
            for result in self.recognition.decode(entry.crop.image):
                self._audit(track_id, entry.crop.frame_idx, result)
                if parsed_is_empty(result.parsed):
                    continue
                self.aggregator.add_observation(
                    TagObservation(
                        frame_idx=entry.crop.frame_idx,
                        timestamp_s=entry.crop.timestamp_s,
                        track_id=track_id,
                        bbox_xyxy=entry.crop.bbox_xyxy,
                        parsed=result.parsed,
                        detection_confidence=entry.crop.detection_confidence,
                        sharpness=entry.crop.sharpness,
                    )
                )

        # Drop the buffer once we have committed observations.
        self.aggregator.clear_crops(track_id)

    def _audit(self, track_id: int, frame_idx: int, result) -> None:
        """Append one JSONL line describing this decoder call. No-op when disabled.

        ``result`` is a ``recognition.RecognitionResult``. Audit keys
        (``backend``/``ocr_confidence``/``raw_text``) are kept stable for
        existing log tooling; ``backend`` now also covers ``qr``/``barcode``.
        """
        if self._audit_path is None:
            return
        parsed = result.parsed
        try:
            row = {
                "track_id": int(track_id),
                "frame_idx": int(frame_idx),
                "backend": result.decoder,
                "ocr_confidence": float(result.confidence),
                "raw_text": result.text,
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
                    "extra_fields": parsed.extra_fields,
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
