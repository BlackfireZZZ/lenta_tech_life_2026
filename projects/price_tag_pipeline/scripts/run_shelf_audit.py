#!/usr/bin/env python3
"""Shelf-audit runner (P2): OUT_OF_STOCK + MISSING_PRICE_TAG alerts.

Pipeline (all detector-only, base.txt deps — NO OCR/VLM needed here):

    video ─► product-facing detector (rotate=ccw, tracked) ─► product traces
          ─► price-tag detector       (rotate=ccw, tracked) ─► tag traces
          ─► associate_tracks (upright-space, persistence gate)
          ─► ShelfAudit{relations, alerts} + evidence crops

Writes ``outputs/shelf_audit/<video>/{audit.json,alerts.json,crops/*.jpg}``.
Never touches the graded 29-column CSV / submission path. Price/name/barcode
on cards + est-lost-revenue are P3 (need the full recognition pipeline).

See docs/shelf-audit.md (P2).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2

from price_tag_pipeline.config import DetectorConfig
from price_tag_pipeline.detector import build_detector, read_video_fps
from price_tag_pipeline.shelf_analytics import (
    AssociationConfig,
    ShelfAlert,
    ShelfAudit,
    TrackTrace,
    associate_tracks,
)
from price_tag_pipeline.shelf_analytics.schema import AlertType

# Checkpoints + videos are .gitignore'd → main working tree only.
MAIN_TREE = Path("E:/Hackatons/lenta_tech_life_2026")
DEFAULT_PRODUCT_W = MAIN_TREE / "data/checkpoints/product_detector/product_facing_retail_pretrain_yolo11s_best.pt"
DEFAULT_TAG_W = MAIN_TREE / "data/checkpoints/detector/best.pt"
VIDEO_DIR = MAIN_TREE / "data/raw/videos"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, default=None,
                   help="single video; omit with --all to run every video in data/raw/videos")
    p.add_argument("--all", action="store_true")
    p.add_argument("--product-weights", type=Path, default=DEFAULT_PRODUCT_W)
    p.add_argument("--tag-weights", type=Path, default=DEFAULT_TAG_W)
    p.add_argument("--product-conf", type=float, default=0.30)  # P0-recommended
    p.add_argument("--tag-conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--rotation", default="ccw")
    p.add_argument("--max-frames", type=int, default=0, help="0 = whole video")
    p.add_argument("--persistence", type=int, default=10,
                   help="min frames an unmatched track must persist to alert")
    p.add_argument("--device", default=None)
    p.add_argument("--outdir", type=Path, default=MAIN_TREE / "outputs/shelf_audit")
    return p.parse_args()


def _cfg(model_path: Path, conf: float, imgsz: int, rotation: str, device) -> DetectorConfig:
    return DetectorConfig(
        backend="yolo", model_path=str(model_path), conf=conf, iou=0.50,
        device=device, classes=None,
        tracker_yaml="configs/trackers/bytetrack.yaml",
        image_size=imgsz, frame_rotation=rotation,
    )


def _collect_traces(cfg: DetectorConfig, video: Path, max_frames: int):
    """Stream the video once → {track_id: {frame_idx: bbox}} + (W, H)."""
    det = build_detector(cfg)
    boxes: dict[int, dict[int, tuple]] = defaultdict(dict)
    frame_w = frame_h = 0
    for frame_idx, (frame, dets) in enumerate(det.stream_video(str(video))):
        if frame_idx == 0:
            frame_h, frame_w = frame.shape[:2]
        for d in dets:
            if d.track_id is None:
                continue
            boxes[d.track_id][frame_idx] = tuple(float(v) for v in d.bbox_xyxy)
        if max_frames and frame_idx + 1 >= max_frames:
            break
    traces = [TrackTrace(track_id=tid, boxes=bx) for tid, bx in boxes.items()]
    return traces, frame_w, frame_h


def _grab_frames(video: Path, wanted: set[int]) -> dict[int, "cv2.Mat"]:
    """Sequentially read the few frames needed for evidence crops."""
    if not wanted:
        return {}
    cap = cv2.VideoCapture(str(video))
    out: dict[int, "cv2.Mat"] = {}
    last = max(wanted)
    idx = -1
    while idx < last:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx in wanted:
            out[idx] = frame
    cap.release()
    return out


def _save_crop(frame, bbox, path: Path, pad: float = 0.06) -> bool:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    x1 = int(max(0, x1 - bw * pad)); y1 = int(max(0, y1 - bh * pad))
    x2 = int(min(w, x2 + bw * pad)); y2 = int(min(h, y2 + bh * pad))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return False
    cv2.imwrite(str(path), frame[y1:y2, x1:x2])
    return True


def _mid_frame(track_boxes_frames: list[int]) -> int:
    fr = sorted(track_boxes_frames)
    return fr[len(fr) // 2]


def audit_video(video: Path, args: argparse.Namespace) -> dict:
    print(f"\n=== {video.name} ===")
    fps = read_video_fps(str(video))

    print("[1/3] product-facing detector ...")
    prod_traces, fw, fh = _collect_traces(
        _cfg(args.product_weights, args.product_conf, args.imgsz, args.rotation, args.device),
        video, args.max_frames)
    print(f"      {len(prod_traces)} product tracks  (frame {fw}x{fh})")

    print("[2/3] price-tag detector ...")
    tag_traces, fw2, fh2 = _collect_traces(
        _cfg(args.tag_weights, args.tag_conf, args.imgsz, args.rotation, args.device),
        video, args.max_frames)
    fw, fh = fw or fw2, fh or fh2
    print(f"      {len(tag_traces)} price-tag tracks")

    print("[3/3] associate + alerts ...")
    cfg = AssociationConfig(rotation=args.rotation, persistence_min_frames=args.persistence)
    res = associate_tracks(tag_traces, prod_traces, frame_w=fw, frame_h=fh,
                           fps=fps, cfg=cfg)

    out_dir = args.outdir / video.stem
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    tag_by_id = {t.track_id: t for t in tag_traces}
    prod_by_id = {p.track_id: p for p in prod_traces}

    # Persistence-gated alert candidates + the frames we must read for crops.
    cands: list[tuple] = []  # (AlertType, UnmatchedTrack, source_trace)
    for u in res.unmatched_price_tags:
        if u.persistent:
            cands.append((AlertType.OUT_OF_STOCK, u, tag_by_id[u.track_id]))
    for u in res.unmatched_products:
        if u.persistent:
            cands.append((AlertType.MISSING_PRICE_TAG, u, prod_by_id[u.track_id]))

    wanted = {_mid_frame(list(src.boxes)) for _t, _u, src in cands}
    frames = _grab_frames(video, wanted)

    alerts: list[ShelfAlert] = []
    for atype, u, src in cands:
        fi = _mid_frame(list(src.boxes))
        crop_rel = None
        frame = frames.get(fi)
        # box at that frame (real) falls back to the median representative box
        bbox = src.boxes.get(fi, u.representative_bbox)
        aid = f"{'oos' if atype is AlertType.OUT_OF_STOCK else 'notag'}_{video.stem}_{u.track_id:04d}"
        if frame is not None and _save_crop(frame, bbox, crops_dir / f"{aid}.jpg"):
            crop_rel = f"crops/{aid}.jpg"
        alerts.append(ShelfAlert(
            id=aid, type=atype, severity="high", video_id=video.stem,
            timestamp_s=fi / (fps if fps > 1e-6 else 1.0), frame_idx=fi,
            bbox_xyxy=bbox, track_id=u.track_id, evidence_crop=crop_rel,
            first_seen_s=u.first_s, last_seen_s=u.last_s,
            persistence_frames=u.n_frames,
        ))

    n_oos = sum(a.type is AlertType.OUT_OF_STOCK for a in alerts)
    n_notag = sum(a.type is AlertType.MISSING_PRICE_TAG for a in alerts)
    n_ok = sum(r.status == "ok" for r in res.relations)
    summary = {
        "n_out_of_stock": n_oos,
        "n_missing_price_tag": n_notag,
        "n_relations_ok": n_ok,
        "n_relations_ambiguous": len(res.relations) - n_ok,
        "n_product_tracks": len(prod_traces),
        "n_price_tag_tracks": len(tag_traces),
        "filtered_non_persistent": (
            sum(not u.persistent for u in res.unmatched_price_tags)
            + sum(not u.persistent for u in res.unmatched_products)
        ),
        "est_lost_revenue_rub": None,  # P3: needs recognized prices
    }
    audit = ShelfAudit(
        video_id=video.stem, fps=fps, summary=summary,
        relations=res.relations, alerts=tuple(alerts),
        metadata={
            "product_weights": str(args.product_weights),
            "tag_weights": str(args.tag_weights),
            "product_conf": args.product_conf, "tag_conf": args.tag_conf,
            "rotation": args.rotation, "max_frames": args.max_frames,
            "persistence_min_frames": args.persistence,
            "note": "detector-only pass; CSV/submission path untouched",
        },
    )
    (out_dir / "audit.json").write_text(
        json.dumps(audit.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "alerts.json").write_text(
        json.dumps([a.to_dict() for a in alerts], ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"      wrote {len(alerts)} alerts -> {out_dir}")
    return {"video": video.stem, **summary}


def main() -> int:
    args = parse_args()
    for w in (args.product_weights, args.tag_weights):
        if not w.exists():
            raise SystemExit(f"Missing checkpoint: {w}")

    if args.all:
        videos = sorted(VIDEO_DIR.glob("*.mp4"))
    elif args.video:
        videos = [args.video]
    else:
        raise SystemExit("Pass --video <path> or --all")
    if not videos:
        raise SystemExit("No videos found")

    rollup = [audit_video(v, args) for v in videos]
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "rollup.json").write_text(
        json.dumps(rollup, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] {len(rollup)} video(s) -> {args.outdir}/rollup.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
