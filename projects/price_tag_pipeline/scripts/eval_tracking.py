#!/usr/bin/env python3
"""Tracking-quality harness — detector + tracker only, NO OCR.

Streams the detector+tracker over each video and reduces the per-frame track
stream to the duplicate/fragmentation numbers in
``price_tag_pipeline.metrics.tracking``. This is the gate for every tracker
change: run it before and after, compare ``dup_ratio``.

Data lives in the MAIN checkout, never the worktree (see docs/data/layout.md),
so the video/GT defaults point there.

Usage (from projects/price_tag_pipeline)::

    ../../.venv/Scripts/python.exe scripts/eval_tracking.py \\
        --config configs/balanced.yaml --max-frames 500 --imgsz 768 \\
        --json-out runs/tracking_baseline.json

CPU note: torch is CPU-only here, so keep ``--max-frames`` modest and use a
smaller ``--imgsz`` for fast iteration; the tracker still sees *consecutive*
frames (never strided — striding would corrupt the very association we measure).
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import logging
import sys
import time
from pathlib import Path

THIS = Path(__file__).resolve().parent
SRC = THIS.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Data is in the main checkout: <repo_root>/data. This file is at
# <repo_root>/.claude/worktrees/tracking/projects/price_tag_pipeline/scripts,
# so the repo root is 6 levels up (parents[5]).
_MAIN_REPO = THIS.parents[5]
DEFAULT_VIDEOS = _MAIN_REPO / "data" / "raw" / "videos"
DEFAULT_GT = _MAIN_REPO / "data" / "raw" / "annotations" / "csv"

LOGGER = logging.getLogger("eval_tracking")


def gt_unique_tag_count(gt_csv: Path) -> int:
    """One row per physically-unique tag (the task's CSV contract)."""
    with gt_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    return max(0, len(rows) - 1)  # minus header


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/balanced.yaml")
    p.add_argument("--videos-dir", default=str(DEFAULT_VIDEOS))
    p.add_argument("--gt-dir", default=str(DEFAULT_GT))
    p.add_argument(
        "--videos",
        nargs="*",
        default=None,
        help="Explicit video stems (default: all *.mp4 with a matching GT csv)",
    )
    p.add_argument("--max-frames", type=int, default=500)
    p.add_argument("--imgsz", type=int, default=None, help="override detector imgsz")
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--tracker", default=None, help="override detector.tracker_yaml (path)"
    )
    p.add_argument(
        "--weights", default=None, help="override detector.model_path (e.g. fine-tuned best.pt)"
    )
    p.add_argument(
        "--rotate", default=None, help="override detector.frame_rotation (ccw|cw|none)"
    )
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from price_tag_pipeline.config import load_config
    from price_tag_pipeline.detector import build_detector, read_video_fps
    from price_tag_pipeline.metrics.tracking import (
        TrackingSuite,
        TrackRecord,
        compute_tracking_metrics,
    )

    cfg = load_config(args.config)
    videos_dir = Path(args.videos_dir)
    gt_dir = Path(args.gt_dir)

    if args.videos:
        stems = list(args.videos)
    else:
        stems = sorted(
            v.stem
            for v in videos_dir.glob("*.mp4")
            if (gt_dir / f"{v.stem}.csv").is_file()
        )
    if not stems:
        LOGGER.error("No videos with matching GT found in %s", videos_dir)
        return 2

    LOGGER.info(
        "config=%s tracker=%s imgsz=%s max_frames=%d videos=%s",
        args.config,
        cfg.detector.tracker_yaml,
        args.imgsz or cfg.detector.image_size,
        args.max_frames,
        ",".join(stems),
    )

    suite = TrackingSuite()
    for stem in stems:
        video_path = videos_dir / f"{stem}.mp4"
        gt_csv = gt_dir / f"{stem}.csv"
        if not video_path.is_file():
            LOGGER.warning("skip %s: video missing", stem)
            continue
        n_gt = gt_unique_tag_count(gt_csv)

        det_cfg = cfg.detector
        if args.imgsz:
            det_cfg = dataclasses.replace(det_cfg, image_size=args.imgsz)
        if args.device:
            det_cfg = dataclasses.replace(det_cfg, device=args.device)
        if args.tracker:
            det_cfg = dataclasses.replace(det_cfg, tracker_yaml=args.tracker)
        if args.weights:
            det_cfg = dataclasses.replace(det_cfg, model_path=args.weights)
        if args.rotate:
            det_cfg = dataclasses.replace(det_cfg, frame_rotation=args.rotate)

        try:
            detector = build_detector(det_cfg)
        except Exception as exc:
            LOGGER.error("skip %s: detector build failed: %s", stem, exc)
            continue

        fps = read_video_fps(str(video_path))
        records: list[TrackRecord] = []
        frame_w = frame_h = 0
        frames = 0
        t0 = time.time()
        try:
            for frame, dets in detector.stream_video(str(video_path)):
                if frames == 0:
                    frame_h, frame_w = frame.shape[:2]
                for d in dets:
                    if d.track_id is None:
                        continue
                    records.append(
                        TrackRecord(
                            frame_idx=frames,
                            track_id=int(d.track_id),
                            bbox_xyxy=tuple(int(v) for v in d.bbox_xyxy),
                            confidence=float(d.confidence),
                        )
                    )
                frames += 1
                if frames >= args.max_frames:
                    break
        except Exception as exc:
            LOGGER.error("error streaming %s after %d frames: %s", stem, frames, exc)
            if not records:
                continue

        dedup_time_window_s = cfg.aggregation.dedup_time_window_s
        report = compute_tracking_metrics(
            video=stem,
            records=records,
            n_gt=n_gt,
            frame_w=frame_w,
            frame_h=frame_h,
            fps=fps,
            frames_processed=frames,
            min_observations=cfg.aggregation.min_observations_per_track,
            dedup_iou=cfg.aggregation.dedup_iou_threshold,
            dedup_time_window_s=dedup_time_window_s,
        )
        suite.add(report)
        LOGGER.info(
            "%s  (%.1fs, %.2f fps proc)",
            report.one_line(),
            time.time() - t0,
            frames / max(1e-6, time.time() - t0),
        )

    summary = suite.summary()
    LOGGER.info("SUMMARY %s", json.dumps(summary, ensure_ascii=False))

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "config": args.config,
                    "tracker_yaml": cfg.detector.tracker_yaml,
                    "imgsz": args.imgsz or cfg.detector.image_size,
                    "max_frames": args.max_frames,
                    "summary": summary,
                    "reports": [r.as_dict() for r in suite.reports],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        LOGGER.info("wrote %s", out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
