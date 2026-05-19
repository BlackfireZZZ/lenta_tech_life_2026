"""Optional per-frame detector trace — a NON-graded QA side-artifact.

The graded pipeline keeps exactly ONE row per unique tag (the locked
29-column CSV). This module collects, *additionally and side-channel only*,
every detector box on every frame, so the product can replay the clip with
the raw detector output overlaid — letting a reviewer judge the **detector
itself**, separate from the final per-tag result.

It never touches ``tags`` / the CSV. The pipeline drives it through the same
optional-observer seam as ``progress``: a default of ``None`` means the
research / CLI path is byte-for-byte unchanged and pays nothing.

Coordinate facts (kept identical to ``backend/app/predictions.py`` so the
detector overlay and the best-frame box use the exact same math):

* The detector un-projects every box back to **original-frame** pixels even
  when it pre-rotates for the model (``detector.unrotate_box_xyxy``), and
  the yielded frame is the original. The browser plays that same raw clip.
  So normalised = pixel / raw-frame-size lines up directly — no rotation
  here.
* All boxes the detector yields are already above ``cfg.detector.conf`` (it
  is passed to YOLO as ``conf=``). ``conf_threshold`` records that value so
  the UI can state it honestly; per-box ``score`` is kept too.

Size is bounded: a long clip is uniformly **strided** down to
``max_frames`` and the artifact is flagged ``sampled=True`` (a frame is kept
or skipped whole, so a kept frame still shows *all* its boxes). This keeps
the JSON a few hundred KB at most.

Everything is guarded: a bad frame is dropped, never raised — the graded run
must always survive (architecture.md §5.2, the "degrade safely" rule).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from .types import Detection

LOGGER = logging.getLogger(__name__)

# A per-frame loop over a multi-minute clip can be thousands of frames; this
# cap keeps the side-artifact small. Beyond it, frames are sampled uniformly
# (whole-frame stride) and the result is flagged ``sampled``.
DEFAULT_MAX_FRAMES = 1800


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


class FrameTraceCollector:
    """Accumulate normalised detector boxes per frame (default-off observer).

    Pass an instance as ``PriceTagPipeline.run(..., on_frame=collector)``.
    The pipeline calls ``collector(frame_idx, timestamp_s, fw, fh, dets)``
    once per frame that has detections; :meth:`to_dict` then yields the
    contract artifact. The collector is self-guarding: any per-frame failure
    is swallowed (logged once) so it can never break the graded pipeline.
    """

    def __init__(
        self,
        conf_threshold: float,
        frames_total: int = 0,
        max_frames: int = DEFAULT_MAX_FRAMES,
    ) -> None:
        self._conf = float(conf_threshold)
        self._max_frames = max(1, int(max_frames))
        # Whole-frame stride so the artifact stays bounded on a long clip.
        # 0/unknown total → keep every frame (degrades to the same cap by
        # construction on the clips this product sees).
        stride = 1
        if frames_total and frames_total > self._max_frames:
            stride = (frames_total + self._max_frames - 1) // self._max_frames
        self._stride = max(1, stride)
        self._sampled = self._stride > 1
        self._frames: list[dict] = []
        self._fw = 0
        self._fh = 0
        self._warned = False

    def __call__(
        self,
        frame_idx: int,
        timestamp_s: float,
        frame_w: int,
        frame_h: int,
        detections: Sequence[Detection],
    ) -> None:
        try:
            if frame_idx % self._stride != 0:
                return
            if frame_w <= 0 or frame_h <= 0 or not detections:
                return
            if not self._fw:
                self._fw, self._fh = int(frame_w), int(frame_h)
            boxes: list[list[float]] = []
            for det in detections:
                x1, y1, x2, y2 = (float(v) for v in det.bbox_xyxy)
                nx1, nx2 = sorted((x1 / frame_w, x2 / frame_w))
                ny1, ny2 = sorted((y1 / frame_h, y2 / frame_h))
                nx1, ny1 = _clamp01(nx1), _clamp01(ny1)
                nx2, ny2 = _clamp01(nx2), _clamp01(ny2)
                if nx2 - nx1 <= 0.0 or ny2 - ny1 <= 0.0:
                    continue  # degenerate after clamping — drop it
                boxes.append([
                    round(nx1, 4),
                    round(ny1, 4),
                    round(nx2, 4),
                    round(ny2, 4),
                    round(float(det.confidence), 3),
                ])
            if boxes:
                self._frames.append(
                    {"t_ms": int(round(float(timestamp_s) * 1000.0)),
                     "boxes": boxes}
                )
        except Exception as exc:  # noqa: BLE001 - never fatal to the run
            if not self._warned:
                LOGGER.warning("frame-trace collect failed (dropped): %s", exc)
                self._warned = True

    def to_dict(self) -> dict:
        """The contract artifact (``ProcessResponse.detections``).

        ``frames`` is time-ordered; each entry is ``{t_ms, boxes}`` with
        ``boxes`` a list of ``[x1, y1, x2, y2, score]`` normalised to [0,1].
        Frames with no detection are omitted (the UI shows no box there).
        """
        return {
            "frame_width": self._fw,
            "frame_height": self._fh,
            "conf_threshold": round(self._conf, 4),
            "sampled": self._sampled,
            "frames": self._frames,
        }

    @property
    def n_frames(self) -> int:
        return len(self._frames)
