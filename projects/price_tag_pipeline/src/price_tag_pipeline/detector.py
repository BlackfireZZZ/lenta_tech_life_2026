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

import numpy as np

from .config import DetectorConfig
from .types import Detection

LOGGER = logging.getLogger(__name__)

DEFAULT_FPS_FALLBACK = 30.0
HF_MODEL_PREFIX = "hf://"

# Names Ultralytics resolves from *its own* bundled cfg/trackers/ when passed
# bare. We ship tuned same-named files under configs/trackers/; the resolver
# below makes those win so the tuned config is not silently ignored.
ULTRALYTICS_BUILTIN_TRACKERS = {"botsort.yaml", "bytetrack.yaml"}


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for video detector runtime. Install "
            "projects/price_tag_pipeline/requirements/base.txt in the project venv."
        ) from exc
    return cv2


def read_video_fps(video_path: str) -> float:
    """Read FPS from the container. Falls back to 30 fps with a loud warning."""
    cv2 = _require_cv2()
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


def read_video_frame_count(video_path: str) -> int:
    """Total frame count from the container, or 0 if unknown.

    Used only to turn the per-frame loop into a 0..1 progress fraction.
    Some containers report 0 / a bogus value for ``CAP_PROP_FRAME_COUNT``;
    in that case we return 0 and progress degrades to phase-only updates
    (an indeterminate bar) — never an exception, never a wrong total.
    """
    cv2 = _require_cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        LOGGER.warning("Could not open %s to read frame count.", video_path)
        return 0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return count if count > 0 else 0


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
        model_path = resolve_detector_model_path(cfg.model_path)
        if not Path(model_path).exists() and "/" in model_path:
            LOGGER.warning(
                "Detector checkpoint '%s' does not exist locally. Ultralytics may "
                "attempt to download it.",
                model_path,
            )
        # Lazy import so the package is importable without ultralytics installed
        # (useful for tests of parser/aggregator on a CI box without GPU deps).
        from ultralytics import YOLO  # type: ignore

        # Resolve eagerly (before the model load / possible download) so a
        # misconfigured tracker fails fast, not silently mid-stream on stock
        # defaults.
        self._tracker_yaml = resolve_tracker_yaml(self.cfg.tracker_yaml)
        LOGGER.info("Tracker config: %s", self._tracker_yaml)
        self._YOLO = YOLO
        self.model = YOLO(model_path)
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
        """Yield (frame_bgr, detections), one upright frame at a time.

        Frames are read with OpenCV and rotated per ``cfg.rotate`` BEFORE
        detection: the Lenta scan-robot camera is mounted 90° clockwise, so
        raw clips are sideways landscape and a detector trained on upright
        tags misses/duplicates badly (this roughly halves recall). We feed the
        model the rotated frame and run a per-frame ``model.track(persist=…)``
        loop so detections, track IDs, crops and bboxes all live in the same
        upright coordinate space. ``model.predict`` + a light IoU tracker is
        the fallback if ``model.track`` raises.
        """
        cv2 = _require_cv2()
        fps = float(fps_override) if fps_override else read_video_fps(video_path)
        rot = {
            "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,
            "cw": cv2.ROTATE_90_CLOCKWISE,
        }.get((self.cfg.rotate or "none").lower().strip())
        LOGGER.info(
            "Using FPS=%.3f rotate=%s for %s", fps, self.cfg.rotate, video_path
        )

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        common = dict(
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            device=self.cfg.device,
            verbose=False,
            imgsz=self.cfg.image_size,
        )
        if self.cfg.classes is not None:
            common["classes"] = self.cfg.classes

        use_predict = False
        next_tid = 1
        fb_tracks: list[tuple[int, tuple[int, int, int, int], int]] = []
        frame_idx = -1
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_idx += 1
                if rot is not None:
                    frame = cv2.rotate(frame, rot)

                if not use_predict:
                    try:
                        # persist=False on frame 0 forces a FRESH tracker even
                        # if this model object was reused for a prior video.
                        result = self.model.track(
                            frame,
                            persist=frame_idx > 0,
                            tracker=self._tracker_yaml,
                            **common,
                        )[0]
                    except Exception as exc:
                        LOGGER.warning(
                            "model.track() failed (%s); predict()+IoU tracker "
                            "for the remaining frames.",
                            exc,
                        )
                        use_predict = True
                if use_predict:
                    result = self.model.predict(frame, **common)[0]

                boxes = getattr(result, "boxes", None)
                names = getattr(result, "names", {}) or {}
                if boxes is None or len(boxes) == 0:
                    yield frame, []
                    continue

                xyxy = boxes.xyxy
                confs = boxes.conf
                classes = boxes.cls
                ids = None if use_predict else getattr(boxes, "id", None)
                fb_ids: list[int] | None = None
                if ids is None:
                    fb_ids, fb_tracks, next_tid = _assign_iou_track_ids(
                        boxes_xyxy=[
                            tuple(int(round(float(v))) for v in row.tolist())
                            for row in xyxy
                        ],
                        tracks=fb_tracks,
                        next_tid=next_tid,
                        frame_idx=frame_idx,
                    )

                detections: list[Detection] = []
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = (int(round(float(v))) for v in xyxy[i].tolist())
                    conf = float(confs[i].item()) if confs is not None else 1.0
                    cls_id = int(classes[i].item()) if classes is not None else 0
                    track_id = int(ids[i].item()) if ids is not None else fb_ids[i]
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
        finally:
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


def resolve_detector_model_path(model_path: str) -> str:
    """Resolve a detector model path understood by runtime configs.

    Supported forms:
    - local path or Ultralytics model name, passed through unchanged;
    - ``hf://owner/repo/path/in/repo.pt``, downloaded through Hugging Face Hub.

    The default production configs use OpenFoodFacts'
    ``hf://openfoodfacts/price-tag-detection/weights/best.pt`` model so a fresh
    checkout has a real detector before we fine-tune our own checkpoint.
    """
    raw = str(model_path).strip()
    if not raw.startswith(HF_MODEL_PREFIX):
        return raw

    spec = raw[len(HF_MODEL_PREFIX):].strip("/")
    parts = spec.split("/", 2)
    if len(parts) != 3 or not all(parts):
        raise ValueError(
            "HF detector URI must look like "
            "hf://owner/repo/path/to/file.pt, got: "
            f"{model_path!r}"
        )
    repo_id = f"{parts[0]}/{parts[1]}"
    filename = parts[2]

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "Detector model_path uses hf:// but huggingface-hub is not installed. "
            "Install projects/price_tag_pipeline/requirements/base.txt or set "
            "detector.model_path to a local checkpoint."
        ) from exc

    return hf_hub_download(repo_id=repo_id, filename=filename)


def _project_trackers_dir() -> Optional[Path]:
    """``projects/price_tag_pipeline/configs/trackers`` derived from this file.

    ``detector.py`` lives at ``<proj>/src/price_tag_pipeline/detector.py`` for
    an editable install, so the project root is ``parents[2]``. Returns ``None``
    when the package is installed non-editably (no sibling ``configs/``), in
    which case resolution falls back to CWD-relative lookup.
    """
    candidate = Path(__file__).resolve().parents[2] / "configs" / "trackers"
    return candidate if candidate.is_dir() else None


def resolve_tracker_yaml(tracker_yaml: str) -> str:
    """Resolve the tracker config to an absolute path Ultralytics will load.

    The bug this fixes: a bare value like ``botsort.yaml`` makes Ultralytics
    load *its own* bundled config and silently ignore the project's tuned
    tracker, so every tracker-tuning change is a no-op. We resolve project /
    relative paths to an absolute path so the tuned config actually takes
    effect.

    Resolution order:
    1. Existing path (absolute, or relative to CWD) → absolute path.
    2. Existing path relative to the project root → absolute path.
    3. Bare ``<name>.yaml`` matching a file under ``configs/trackers/`` → that
       tuned file (so a profile that still says just ``botsort.yaml`` upgrades
       to the tuned one instead of Ultralytics' stock defaults).
    4. A bare Ultralytics builtin with no project match → passed through with a
       loud warning (stock, untuned — a tuning regression worth surfacing).
    5. Anything else missing → ``FileNotFoundError`` (fail fast, not silently
       fall back to stock).
    """
    raw = str(tracker_yaml).strip()
    if not raw:
        raise ValueError("detector.tracker_yaml is empty")

    p = Path(raw).expanduser()
    if p.is_file():
        return str(p.resolve())

    project_root = Path(__file__).resolve().parents[2]
    rel_to_root = project_root / raw
    if rel_to_root.is_file():
        return str(rel_to_root.resolve())

    trackers_dir = _project_trackers_dir()
    if trackers_dir is not None:
        by_name = trackers_dir / Path(raw).name
        if by_name.is_file():
            return str(by_name.resolve())

    if raw in ULTRALYTICS_BUILTIN_TRACKERS:
        LOGGER.warning(
            "tracker_yaml=%r resolved to Ultralytics' STOCK bundled config "
            "(no matching file under configs/trackers/). The tracker is "
            "running UNTUNED for this moving-camera scenario. Point "
            "detector.tracker_yaml at configs/trackers/%s.",
            raw, raw,
        )
        return raw

    raise FileNotFoundError(
        f"tracker_yaml={raw!r} not found (looked at CWD, project root, and "
        f"configs/trackers/). Use a path under configs/trackers/ or a builtin "
        f"name ({sorted(ULTRALYTICS_BUILTIN_TRACKERS)})."
    )


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
