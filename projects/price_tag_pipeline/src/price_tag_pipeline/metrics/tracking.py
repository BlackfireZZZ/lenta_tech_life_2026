"""Tracking-quality metrics — no recognition, no OCR.

The hackathon metric punishes duplicates: one physical tag emitted N times →
one row matches GT, the other N-1 are unmatched noise (docs/index.md fact 3).
The tracker is what collapses a tag's many frames into ONE row, so its quality
is measured here as: how close is the number of emitted tracks to the number
of physically-unique GT tags, and how fragmented are the tracks.

Everything here is pure (numpy only) and needs no model — a script feeds it
the per-frame ``(frame_idx, track_id, bbox, conf)`` stream produced by the
detector+tracker, plus the GT unique-tag count for the video.

Key numbers
-----------
- ``dup_ratio`` = qualified_tracks / n_gt. Ideal ≈ 1.0. ``> 1`` means the
  tracker fragments / ID-switches / over-detects (the metric killer). ``< 1``
  means misses or over-merge.
- ``geom_dedup_dup_ratio`` = an UPPER BOUND on what the current bbox-IoU
  cross-track dedup could collapse to (content gate dropped, so it can only
  merge *more* than the real dedup). If this stays ≈ ``dup_ratio`` on the
  moving-camera videos, the IoU-based dedup is structurally unable to fix
  fragmentation → motivates content/barcode-keyed dedup.
- ``border_touch_frac`` = fraction of detections whose box touches a frame
  edge (a tag entering/leaving = motion-blurred + truncated). High values
  motivate border-penalised best-frame selection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class TrackRecord:
    """One detection row: a box for ``track_id`` at ``frame_idx``."""

    frame_idx: int
    track_id: int
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float


@dataclass
class TrackingReport:
    video: str
    n_gt: int
    frames_processed: int
    fps: float

    n_detections: int
    n_raw_tracks: int
    n_qualified_tracks: int  # tracks with >= min_observations frames
    dup_ratio: float  # qualified / n_gt  (≈1 ideal, >1 = fragmentation)

    n_singleton_tracks: int  # exactly 1 frame
    n_short_tracks: int  # < min_observations frames
    track_len_median: float
    track_len_p90: float
    track_len_max: int

    border_touch_frac: float

    geom_dedup_survivors: int  # qualified after geom-only (upper-bound) merge
    geom_dedup_dup_ratio: float

    min_observations: int = 0
    dedup_iou: float = 0.0
    dedup_time_window_s: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)

    def one_line(self) -> str:
        return (
            f"{self.video:<14} gt={self.n_gt:<4} "
            f"raw={self.n_raw_tracks:<4} qual={self.n_qualified_tracks:<4} "
            f"dup={self.dup_ratio:>5.2f}  "
            f"geomdup={self.geom_dedup_dup_ratio:>5.2f}  "
            f"sing/short={self.n_singleton_tracks}/{self.n_short_tracks}  "
            f"len(med/p90/max)={self.track_len_median:.0f}/"
            f"{self.track_len_p90:.0f}/{self.track_len_max}  "
            f"border={self.border_touch_frac:.0%}  "
            f"frames={self.frames_processed}"
        )


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _box_touches_border(
    box: tuple[int, int, int, int], w: int, h: int, eps_x: float, eps_y: float
) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= eps_x or y1 <= eps_y or x2 >= w - eps_x or y2 >= h - eps_y


def compute_tracking_metrics(
    *,
    video: str,
    records: list[TrackRecord],
    n_gt: int,
    frame_w: int,
    frame_h: int,
    fps: float,
    frames_processed: int,
    min_observations: int,
    dedup_iou: float,
    dedup_time_window_s: float,
) -> TrackingReport:
    """Reduce a per-frame detection stream to a tracking-quality report."""
    by_track: dict[int, list[TrackRecord]] = {}
    for r in records:
        by_track.setdefault(r.track_id, []).append(r)

    eps_x = max(2.0, 0.005 * frame_w)
    eps_y = max(2.0, 0.005 * frame_h)
    border_hits = sum(
        1
        for r in records
        if _box_touches_border(r.bbox_xyxy, frame_w, frame_h, eps_x, eps_y)
    )

    lengths: list[int] = []
    qualified: list[int] = []  # track_ids
    n_singleton = 0
    n_short = 0
    for tid, rs in by_track.items():
        n = len(rs)
        lengths.append(n)
        if n == 1:
            n_singleton += 1
        if n < min_observations:
            n_short += 1
        else:
            qualified.append(tid)

    len_arr = np.array(lengths) if lengths else np.array([0])
    n_gt_safe = max(1, n_gt)

    # Geometry-only dedup upper bound: merge qualified tracks if their last
    # boxes overlap >= dedup_iou AND their frame spans are within the time
    # window. No content gate (we have none) → this MERGES AT LEAST AS MUCH as
    # the real content-gated dedup, so it is an optimistic upper bound on what
    # bbox-IoU dedup can ever do.
    win_frames = dedup_time_window_s * fps if fps > 0 else 0.0
    spans: list[tuple[int, int, int, tuple[int, int, int, int]]] = []
    for tid in qualified:
        rs = sorted(by_track[tid], key=lambda r: r.frame_idx)
        spans.append((tid, rs[0].frame_idx, rs[-1].frame_idx, rs[-1].bbox_xyxy))
    spans.sort(key=lambda s: s[1])
    parent = {s[0]: s[0] for s in spans}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(spans)):
        ti, _, ei, bi = spans[i]
        for j in range(i + 1, len(spans)):
            tj, sj, _, bj = spans[j]
            if win_frames and (sj - ei) > win_frames:
                break
            if _iou(bi, bj) >= dedup_iou:
                parent[find(tj)] = find(ti)
    survivors = len({find(t) for t in parent})

    return TrackingReport(
        video=video,
        n_gt=n_gt,
        frames_processed=frames_processed,
        fps=round(fps, 3),
        n_detections=len(records),
        n_raw_tracks=len(by_track),
        n_qualified_tracks=len(qualified),
        dup_ratio=round(len(qualified) / n_gt_safe, 3),
        n_singleton_tracks=n_singleton,
        n_short_tracks=n_short,
        track_len_median=float(np.median(len_arr)),
        track_len_p90=float(np.percentile(len_arr, 90)),
        track_len_max=int(len_arr.max()),
        border_touch_frac=round(border_hits / max(1, len(records)), 4),
        geom_dedup_survivors=survivors,
        geom_dedup_dup_ratio=round(survivors / n_gt_safe, 3),
        min_observations=min_observations,
        dedup_iou=dedup_iou,
        dedup_time_window_s=dedup_time_window_s,
    )


@dataclass
class TrackingSuite:
    """Aggregate of per-video reports for a whole run."""

    reports: list[TrackingReport] = field(default_factory=list)

    def add(self, r: TrackingReport) -> None:
        self.reports.append(r)

    def summary(self) -> dict:
        if not self.reports:
            return {}
        dup = np.array([r.dup_ratio for r in self.reports])
        geom = np.array([r.geom_dedup_dup_ratio for r in self.reports])
        return {
            "videos": len(self.reports),
            "mean_dup_ratio": round(float(dup.mean()), 3),
            "worst_dup_ratio": round(float(dup.max()), 3),
            "mean_geom_dedup_dup_ratio": round(float(geom.mean()), 3),
            "total_gt": sum(r.n_gt for r in self.reports),
            "total_qualified_tracks": sum(
                r.n_qualified_tracks for r in self.reports
            ),
        }
