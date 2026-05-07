from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from ultralytics import YOLO

from .config import DetectorConfig
from .types import Detection


class YoloTrackerDetector:
    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        self.model = YOLO(self.cfg.model_path)

    def stream_video(self, video_path: str) -> Iterator[tuple[np.ndarray, list[Detection]]]:
        results = self.model.track(
            source=video_path,
            stream=True,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            tracker=self.cfg.tracker_yaml,
            classes=self.cfg.classes,
            device=self.cfg.device,
            verbose=False,
            persist=True,
        )
        fps = 30.0
        for frame_idx, result in enumerate(results):
            frame = result.orig_img.copy()
            boxes = result.boxes
            if hasattr(result, "speed"):
                pass
            if boxes is None or len(boxes) == 0:
                yield frame, []
                continue

            ids = getattr(boxes, "id", None)
            names = result.names or {}
            detections: list[Detection] = []
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].tolist()
                x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
                conf = float(boxes.conf[i].item()) if boxes.conf is not None else 1.0
                cls_id = int(boxes.cls[i].item()) if boxes.cls is not None else 0
                track_id = int(ids[i].item()) if ids is not None else None
                class_name = str(names.get(cls_id, cls_id))
                detections.append(
                    Detection(
                        frame_idx=frame_idx,
                        timestamp_s=frame_idx / fps,
                        bbox_xyxy=(x1, y1, x2, y2),
                        confidence=conf,
                        class_id=cls_id,
                        class_name=class_name,
                        track_id=track_id,
                    )
                )
            yield frame, detections

