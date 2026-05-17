"""Detector backends.

`BaseDetector` defines the streaming contract used by the pipeline.

Two implementations:
- `YOLOTrackerDetector` — Ultralytics-native (YOLO11/12/26 + ByteTrack/BoT-SORT)
- `RFDETRDetector` — Roboflow's RF-DETR. Tracking is bolted on separately because
  RF-DETR has no built-in tracker. Disabled by default until the dependency is added.

FPS is read from the container via OpenCV and used for timestamp computation.
The old hardcoded `fps = 30.0` is gone.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import DetectorConfig
from .types import Detection

LOGGER = logging.getLogger(__name__)

DEFAULT_FPS_FALLBACK = 30.0


def read_video_fps(video_path: str) -> float:
    """Read FPS from the container. Falls back to 30 fps with a loud warning."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        LOGGER.warning("Could not open %s to read FPS; falling back to %.1f", video_path, DEFAULT_FPS_FALLBACK)
        return DEFAULT_FPS_FALLBACK
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    if fps <= 1.0 or fps != fps:  # NaN check
        LOGGER.warning("Suspicious FPS %.3f for %s; falling back to %.1f", fps, video_path, DEFAULT_FPS_FALLBACK)
        return DEFAULT_FPS_FALLBACK
    return fps


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class BaseDetector(ABC):
    @abstractmethod
    def stream_video(
        self,
        video_path: str,
        fps_override: Optional[float] = None,
    ) -> Iterator[tuple[np.ndarray, list[Detection]]]:
        """Yield (frame_bgr, detections) one frame at a time."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Ultralytics-native
# ---------------------------------------------------------------------------

class YOLOTrackerDetector(BaseDetector):
    """Wrapper around Ultralytics' YOLO.track().

    Works with any Ultralytics-shipped model (YOLO11/12/26, RT-DETR via
    ultralytics's RT-DETR class — handled separately if needed).
    """

    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        if not Path(cfg.model_path).exists() and not cfg.model_path.endswith(".pt"):
            LOGGER.warning(
                "Detector checkpoint '%s' does not exist locally. Ultralytics may "
                "attempt to download it.",
                cfg.model_path,
            )
        # Lazy import so the package is importable without ultralytics installed
        # (useful for tests of parser/aggregator on a CI box without GPU deps).
        from ultralytics import YOLO  # type: ignore

        self._YOLO = YOLO
        self.model = YOLO(self.cfg.model_path)
        if self.cfg.open_vocab_labels:
            if hasattr(self.model, "set_classes"):
                self.model.set_classes(list(self.cfg.open_vocab_labels))
                LOGGER.info("Open-vocabulary labels set: %s", ", ".join(self.cfg.open_vocab_labels))
            else:
                LOGGER.warning(
                    "Detector does not expose set_classes(); open_vocab_labels ignored: %s",
                    self.cfg.open_vocab_labels,
                )

    def stream_video(
        self,
        video_path: str,
        fps_override: Optional[float] = None,
    ) -> Iterator[tuple[np.ndarray, list[Detection]]]:
        fps = float(fps_override) if fps_override else read_video_fps(video_path)
        LOGGER.info("Using FPS=%.3f for %s", fps, video_path)

        track_kwargs = dict(
            source=video_path,
            stream=True,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            tracker=self.cfg.tracker_yaml,
            device=self.cfg.device,
            verbose=False,
            persist=True,
            imgsz=self.cfg.image_size,
        )
        if self.cfg.classes is not None:
            track_kwargs["classes"] = self.cfg.classes

        try:
            results = self.model.track(**track_kwargs)
            next_tid = 1
            fallback_tracks: list[tuple[int, tuple[int, int, int, int], int]] = []
            for frame_idx, result in enumerate(results):
                frame = result.orig_img  # do not copy — downstream rectifier copies its slice
                boxes = getattr(result, "boxes", None)
                names = getattr(result, "names", {}) or {}

                if boxes is None or len(boxes) == 0:
                    yield frame, []
                    continue

                ids = getattr(boxes, "id", None)
                xyxy = boxes.xyxy
                confs = boxes.conf
                classes = boxes.cls
                fallback_ids: list[int] | None = None
                if ids is None:
                    fallback_ids, fallback_tracks, next_tid = _assign_iou_track_ids(
                        boxes_xyxy=[tuple(int(round(float(v))) for v in row.tolist()) for row in xyxy],
                        tracks=fallback_tracks,
                        next_tid=next_tid,
                        frame_idx=frame_idx,
                    )

                detections: list[Detection] = []
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = (int(round(float(v))) for v in xyxy[i].tolist())
                    conf = float(confs[i].item()) if confs is not None else 1.0
                    cls_id = int(classes[i].item()) if classes is not None else 0
                    track_id = int(ids[i].item()) if ids is not None else fallback_ids[i]
                    detections.append(
                        Detection(
                            frame_idx=frame_idx,
                            timestamp_s=frame_idx / fps,
                            bbox_xyxy=(x1, y1, x2, y2),
                            confidence=conf,
                            class_id=cls_id,
                            class_name=str(names.get(cls_id, cls_id)),
                            track_id=track_id,
                        )
                    )
                yield frame, detections
            return
        except Exception as exc:
            LOGGER.warning(
                "model.track() failed (%s). Falling back to predict()+lightweight IoU tracker.",
                exc,
            )

        yield from self._stream_with_predict_fallback(video_path=video_path, fps=fps)

    def _stream_with_predict_fallback(
        self,
        video_path: str,
        fps: float,
    ) -> Iterator[tuple[np.ndarray, list[Detection]]]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        next_tid = 1
        # [tid, bbox, last_seen_frame]
        tracks: list[tuple[int, tuple[int, int, int, int], int]] = []
        max_age = 15
        iou_gate = 0.3

        frame_idx = -1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            pred = self.model.predict(
                source=frame,
                conf=self.cfg.conf,
                iou=self.cfg.iou,
                device=self.cfg.device,
                verbose=False,
                imgsz=self.cfg.image_size,
            )
            result = pred[0]
            boxes = getattr(result, "boxes", None)
            names = getattr(result, "names", {}) or {}
            detections: list[Detection] = []
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy
                confs = boxes.conf
                classes = boxes.cls
                assigned: set[int] = set()
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = (int(round(float(v))) for v in xyxy[i].tolist())
                    bbox = (x1, y1, x2, y2)
                    conf = float(confs[i].item()) if confs is not None else 1.0
                    cls_id = int(classes[i].item()) if classes is not None else 0

                    best_j = -1
                    best_iou = 0.0
                    for j, (tid, tb, last_seen) in enumerate(tracks):
                        if frame_idx - last_seen > max_age or j in assigned:
                            continue
                        iou = _bbox_iou(bbox, tb)
                        if iou > best_iou:
                            best_iou = iou
                            best_j = j
                    if best_j >= 0 and best_iou >= iou_gate:
                        tid, _, _ = tracks[best_j]
                        tracks[best_j] = (tid, bbox, frame_idx)
                        assigned.add(best_j)
                        track_id = tid
                    else:
                        track_id = next_tid
                        next_tid += 1
                        tracks.append((track_id, bbox, frame_idx))
                        assigned.add(len(tracks) - 1)

                    detections.append(
                        Detection(
                            frame_idx=frame_idx,
                            timestamp_s=frame_idx / fps,
                            bbox_xyxy=bbox,
                            confidence=conf,
                            class_id=cls_id,
                            class_name=str(names.get(cls_id, cls_id)),
                            track_id=track_id,
                        )
                    )
            tracks = [t for t in tracks if frame_idx - t[2] <= max_age]
            yield frame, detections

        cap.release()


def _assign_iou_track_ids(
    boxes_xyxy: list[tuple[int, int, int, int]],
    tracks: list[tuple[int, tuple[int, int, int, int], int]],
    next_tid: int,
    frame_idx: int,
    max_age: int = 15,
    iou_gate: float = 0.3,
) -> tuple[list[int], list[tuple[int, tuple[int, int, int, int], int]], int]:
    """Assign stable-enough IDs when Ultralytics returns boxes without tracker IDs."""
    out: list[int] = []
    assigned: set[int] = set()
    for bbox in boxes_xyxy:
        best_j = -1
        best_iou = 0.0
        for j, (tid, tb, last_seen) in enumerate(tracks):
            if frame_idx - last_seen > max_age or j in assigned:
                continue
            iou = _bbox_iou(bbox, tb)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0 and best_iou >= iou_gate:
            tid, _, _ = tracks[best_j]
            tracks[best_j] = (tid, bbox, frame_idx)
            assigned.add(best_j)
            out.append(tid)
        else:
            tid = next_tid
            next_tid += 1
            tracks.append((tid, bbox, frame_idx))
            assigned.add(len(tracks) - 1)
            out.append(tid)
    tracks = [t for t in tracks if frame_idx - t[2] <= max_age]
    return out, tracks, next_tid


# ---------------------------------------------------------------------------
# RF-DETR (Roboflow) — stub
# ---------------------------------------------------------------------------

class RFDETRDetector(BaseDetector):
    """RF-DETR backend. NOT YET WIRED — installs and inference path land in Stage 3.5.

    Plan:
    - Install: `pip install rfdetr`
    - Load: `from rfdetr import RFDETRBase; model = RFDETRBase(pretrain=cfg.model_path)`
    - Per-frame inference: `model.predict(frame_rgb)`
    - Tracking: wrap with a standalone BoT-SORT instance (boxmot package).

    The pipeline auto-selects between this and YOLOTrackerDetector via
    `DetectorConfig.backend`. Until that's implemented, instantiating this
    raises a clear error so misconfiguration fails fast.
    """

    def __init__(self, cfg: DetectorConfig):
        raise NotImplementedError(
            "RF-DETR backend is not yet wired. Either: "
            "(a) set detector.backend='yolo' in your config to use the Ultralytics path, or "
            "(b) implement RFDETRDetector.stream_video() once `rfdetr` is installed. "
            "See docs/strategy.md §2.1 for the migration plan."
        )

    def stream_video(self, video_path: str, fps_override: Optional[float] = None):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_detector(cfg: DetectorConfig) -> BaseDetector:
    backend = cfg.backend.lower().strip()
    if backend in {"yolo", "ultralytics", "yolo11", "yolo26", "yolo_world", "yolo-world"}:
        return YOLOTrackerDetector(cfg)
    if backend in {"rfdetr", "rf-detr", "rf_detr"}:
        return RFDETRDetector(cfg)
    raise ValueError(f"Unsupported detector backend: {cfg.backend}")


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / max(1, area_a + area_b - inter)
