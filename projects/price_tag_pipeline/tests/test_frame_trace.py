"""FrameTraceCollector — the non-graded per-frame detector-trace artifact.

Pure: no model, no video, no GPU. Pins the contract the gateway + SPA rely
on (normalised [0,1] boxes, time-ordered frames, bounded size) and the
"never break the graded run" guard.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from price_tag_pipeline.frame_trace import FrameTraceCollector  # noqa: E402
from price_tag_pipeline.types import Detection  # noqa: E402


def _det(x1, y1, x2, y2, conf=0.9, frame_idx=0, ts=0.0) -> Detection:
    return Detection(
        frame_idx=frame_idx,
        timestamp_s=ts,
        bbox_xyxy=(x1, y1, x2, y2),
        confidence=conf,
        class_id=0,
        class_name="price_tag",
        track_id=1,
    )


def test_boxes_are_normalised_clamped_and_time_ordered() -> None:
    c = FrameTraceCollector(conf_threshold=0.25)
    # frame 1000x500; one in-bounds box, one spilling past the edge.
    c(0, 0.0, 1000, 500, [_det(100, 50, 300, 250, conf=0.81)])
    c(1, 0.5, 1000, 500, [_det(-20, -10, 1200, 600, conf=0.4)])

    d = c.to_dict()
    assert d["frame_width"] == 1000 and d["frame_height"] == 500
    assert d["conf_threshold"] == 0.25
    assert d["sampled"] is False
    assert [f["t_ms"] for f in d["frames"]] == [0, 500]  # time-ordered ms

    x1, y1, x2, y2, score = d["frames"][0]["boxes"][0]
    assert (x1, y1, x2, y2) == (0.1, 0.1, 0.3, 0.5)
    assert score == 0.81
    # Clamped to the unit square.
    cx1, cy1, cx2, cy2, _ = d["frames"][1]["boxes"][0]
    assert (cx1, cy1, cx2, cy2) == (0.0, 0.0, 1.0, 1.0)


def test_empty_or_degenerate_frames_are_omitted() -> None:
    c = FrameTraceCollector(conf_threshold=0.3)
    c(0, 0.0, 800, 600, [])  # no detections → frame skipped
    c(1, 0.1, 800, 600, [_det(10, 10, 10, 400)])  # zero width → dropped
    c(2, 0.2, 0, 0, [_det(1, 1, 2, 2)])  # bad frame dims → skipped
    assert c.to_dict()["frames"] == []
    assert c.n_frames == 0


def test_long_clip_is_strided_and_flagged_sampled() -> None:
    # 10 frames, cap 4 → stride 3 (ceil(10/4)); only idx 0,3,6,9 kept.
    c = FrameTraceCollector(conf_threshold=0.2, frames_total=10, max_frames=4)
    for i in range(10):
        c(i, i / 10.0, 100, 100, [_det(10, 10, 20, 20, frame_idx=i)])
    d = c.to_dict()
    assert d["sampled"] is True
    assert [f["t_ms"] for f in d["frames"]] == [0, 300, 600, 900]


def test_collector_never_raises_on_a_bad_frame() -> None:
    # A malformed detection must be swallowed (the graded run must survive),
    # and good frames before/after it must still be recorded.
    c = FrameTraceCollector(conf_threshold=0.25)
    c(0, 0.0, 100, 100, [_det(10, 10, 50, 50)])

    class _Bad:
        bbox_xyxy = ("x", "y", "z", "w")  # not numeric → raises inside
        confidence = 0.9

    c(1, 0.1, 100, 100, [_Bad()])  # type: ignore[list-item]
    c(2, 0.2, 100, 100, [_det(20, 20, 60, 60)])

    times = [f["t_ms"] for f in c.to_dict()["frames"]]
    assert times == [0, 200]  # the bad frame dropped, not fatal
