#!/usr/bin/env python3
"""Sweep BoT-SORT knobs on real footage and rank by fragmentation.

Standalone tuning tool — production code is untouched. The detector, imgsz,
videos and frame budget are held FIXED, so any change in track structure
across candidates is attributable to the *tracker config*, not the detector.

A FRESH detector is built per candidate via the production ``build_detector``
path: Ultralytics binds the tracker on the first ``model.track`` call and
ignores later ``tracker=`` args on the same model object, so reusing one model
across configs silently measures the same tracker N times. A new detector per
config (re-resolves the tracker yaml) is the only correct way — weight reload
is ~2 s on GPU, negligible vs per-config inference.

Objective (no per-frame MOT GT exists by design — a proxy, not MOTA/IDF1):
for identical detections, a better tracker creates fewer spurious / fragmented
tracks. Minimise singleton+short tracks (ID churn) then raw count, GUARDED by
keeping qualified-track coverage near its plateau so a config cannot "win" by
over-merging distinct tags into a few blobs (under-counting = metric loss).

Usage::

    .venv/Scripts/python.exe scripts/sweep_trackers.py --device 0 \\
        --imgsz 1280 --videos 26_12-20 25_12-20 --json-out runs/sweep.json
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import logging
import sys
import tempfile
import time
from pathlib import Path

import yaml

THIS = Path(__file__).resolve().parent
SRC = THIS.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

_MAIN_REPO = THIS.parents[5]
DEFAULT_VIDEOS = _MAIN_REPO / "data" / "raw" / "videos"
DEFAULT_GT = _MAIN_REPO / "data" / "raw" / "annotations" / "csv"
DEFAULT_STEMS = ["26_12-20", "25_12-20"]  # the fragmentation-heavy + rich ones

LOGGER = logging.getLogger("sweep")


def _gt_count(p: Path) -> int:
    with p.open("r", encoding="utf-8", newline="") as f:
        return max(0, len(list(csv.reader(f))) - 1)


def _candidates() -> list[tuple[str, dict]]:
    """Grid aimed at the UNDER-segmentation regime exposed by the
    orientation fix: with good weights + upright frames the tracker glues one
    ID across many tags as the shelf pans (12 tracks for 71 GT tags, a
    1747-frame track on an 89 s clip). We want tracks to END when a tag
    leaves and a NEW id when the next enters — i.e. less sticky: smaller
    track_buffer, higher new_track_thresh, lower match_thresh.
    """
    grid: list[tuple[str, dict]] = [("base", {})]
    for v in (5, 10, 15, 20, 30):
        grid.append((f"buffer{v}", {"track_buffer": v}))
    for v in (0.70, 0.80, 0.90):
        grid.append((f"new{v}", {"new_track_thresh": v}))
    for v in (0.50, 0.60, 0.70):
        grid.append((f"match{v}", {"match_thresh": v}))
    for v in (0.60, 0.75):
        grid.append((f"high{v}", {"track_high_thresh": v}))
    grid.append(("split-aggr",
                 {"track_buffer": 10, "new_track_thresh": 0.8,
                  "match_thresh": 0.6}))
    grid.append(("split-mid",
                 {"track_buffer": 15, "new_track_thresh": 0.75,
                  "match_thresh": 0.7}))
    return grid


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/balanced.yaml")
    p.add_argument("--base-tracker", default="configs/trackers/botsort.yaml")
    p.add_argument("--videos", nargs="*", default=DEFAULT_STEMS)
    p.add_argument("--videos-dir", default=str(DEFAULT_VIDEOS))
    p.add_argument("--gt-dir", default=str(DEFAULT_GT))
    p.add_argument("--max-frames", type=int, default=1200)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--device", default="0")
    p.add_argument("--weights", default=None, help="override detector.model_path")
    p.add_argument("--json-out", default="runs/sweep.json")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from price_tag_pipeline.config import load_config
    from price_tag_pipeline.detector import build_detector
    from price_tag_pipeline.metrics.tracking import (
        TrackRecord,
        compute_tracking_metrics,
    )

    cfg = load_config(args.config)
    base_tracker = yaml.safe_load(Path(args.base_tracker).read_text(encoding="utf-8"))
    videos_dir, gt_dir = Path(args.videos_dir), Path(args.gt_dir)
    stems = args.videos
    gt = {s: _gt_count(gt_dir / f"{s}.csv") for s in stems}

    LOGGER.info(
        "model=%s imgsz=%d device=%s videos=%s frames<=%d",
        cfg.detector.model_path, args.imgsz, args.device,
        ",".join(stems), args.max_frames,
    )

    tmpdir = Path(tempfile.mkdtemp(prefix="trk_sweep_"))
    results: list[dict] = []
    for name, override in _candidates():
        merged = dict(base_tracker)
        merged.update(override)
        tcfg = tmpdir / f"{name}.yaml"
        tcfg.write_text(yaml.safe_dump(merged, sort_keys=False), encoding="utf-8")

        det_cfg = dataclasses.replace(
            cfg.detector,
            tracker_yaml=str(tcfg),
            image_size=args.imgsz,
            device=args.device,
            **({"model_path": args.weights} if args.weights else {}),
        )
        agg = {"raw": 0, "qual": 0, "sing_short": 0, "gt": 0}
        per_video: dict[str, dict] = {}
        t0 = time.time()
        # Fresh detector per config → tracker yaml is actually (re)applied.
        detector = build_detector(det_cfg)
        for stem in stems:
            vp = videos_dir / f"{stem}.mp4"
            records: list[TrackRecord] = []
            fw = fh = 0
            frames = 0
            for frame, dets in detector.stream_video(str(vp)):
                if frames == 0:
                    fh, fw = frame.shape[:2]
                for d in dets:
                    if d.track_id is None:
                        continue
                    records.append(
                        TrackRecord(
                            frames, int(d.track_id),
                            tuple(int(v) for v in d.bbox_xyxy),
                            float(d.confidence),
                        )
                    )
                frames += 1
                if frames >= args.max_frames:
                    break
            rep = compute_tracking_metrics(
                video=stem, records=records, n_gt=gt[stem],
                frame_w=fw, frame_h=fh, fps=20.0, frames_processed=frames,
                min_observations=cfg.aggregation.min_observations_per_track,
                dedup_iou=cfg.aggregation.dedup_iou_threshold,
                dedup_time_window_s=cfg.aggregation.dedup_time_window_s,
            )
            agg["raw"] += rep.n_raw_tracks
            agg["qual"] += rep.n_qualified_tracks
            agg["sing_short"] += rep.n_singleton_tracks + rep.n_short_tracks
            agg["gt"] += rep.n_gt
            per_video[stem] = {
                "raw": rep.n_raw_tracks, "qual": rep.n_qualified_tracks,
                "dup": rep.dup_ratio, "sing_short":
                    rep.n_singleton_tracks + rep.n_short_tracks,
            }
        row = {"name": name, "override": override, **agg,
               "secs": round(time.time() - t0, 1), "per_video": per_video}
        results.append(row)
        row["gap"] = abs(agg["qual"] - agg["gt"])
        LOGGER.info(
            "%-12s raw=%-4d qual=%-4d gap=%-4d sing+short=%-4d gt=%-4d (%.0fs) %s",
            name, agg["raw"], agg["qual"], row["gap"], agg["sing_short"],
            agg["gt"], row["secs"], override,
        )

    # Objective: one track per physical tag → qualified ≈ n_gt. The
    # orientation fix flipped the failure from over- to UNDER-segmentation
    # (sticky IDs gluing many tags into a few mega-tracks), so we minimise
    # |qualified − n_gt| (the gap to one-track-per-tag), tie-break by fewer
    # singleton/short tracks (don't trade gluing for pure churn). Proxy, not
    # MOTA — detector recall caps the achievable qual, so the gap floors at
    # whatever recall allows; we pick the config that gets CLOSEST.
    ranked = sorted(results, key=lambda r: (r["gap"], r["sing_short"]))
    LOGGER.info("\n=== RANK by |qual - n_gt| (target gap → 0) ===")
    for r in ranked[:8]:
        LOGGER.info("  %-12s gap=%-4d qual=%-4d raw=%-4d sing+short=%-4d %s",
                    r["name"], r["gap"], r["qual"], r["raw"],
                    r["sing_short"], r["override"])
    best = ranked[0] if ranked else results[0]
    LOGGER.info("WINNER: %s %s (qual=%d vs gt=%d)",
                best["name"], best["override"], best["qual"], best["gt"])

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"base_tracker": base_tracker, "winner": best["name"],
         "winner_override": best["override"], "results": results},
        ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
