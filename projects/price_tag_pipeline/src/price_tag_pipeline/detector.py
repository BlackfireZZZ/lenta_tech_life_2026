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

        results = self.model.track(**track_kwargs)

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

            detections: list[Detection] = []
            for i in range(len(boxes)):
                x1, y1, x2, y2 = (int(round(float(v))) for v in xyxy[i].tolist())
                conf = float(confs[i].item()) if confs is not None else 1.0
                cls_id = int(classes[i].item()) if classes is not None else 0
                track_id = int(ids[i].item()) if ids is not None else None
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
            "See STRATEGY.md §2.1 for the migration plan."
        )

    def stream_video(self, video_path: str, fps_override: Optional[float] = None):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_detector(cfg: DetectorConfig) -> BaseDetector:
    backend = cfg.backend.lower().strip()
    if backend in {"yolo", "ultralytics", "yolo11", "yolo26"}:
        return YOLOTrackerDetector(cfg)
    if backend in {"rfdetr", "rf-detr", "rf_detr"}:
        return RFDETRDetector(cfg)
    raise ValueError(f"Unsupported detector backend: {cfg.backend}")
