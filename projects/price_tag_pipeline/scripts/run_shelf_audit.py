#!/usr/bin/env python3
"""Shelf-audit runner (P2 + base-only P3): OOS / MISSING_PRICE_TAG + cards.

    video ─► product-facing detector (rotate=ccw, tracked) ─► product traces
          ─► price-tag detector       (rotate=ccw, tracked) ─► tag traces
          ─► associate_tracks (upright space, persistence gate)
          ─► build_card_set: sibling facings → one product card
                              (geometry + colour-hist appearance)
          ─► alerts: OUT_OF_STOCK (per tag) + MISSING_PRICE_TAG (per card,
                     collapses the facing-level over-alerting)
          ─► ShelfAudit{relations, alerts, cards} + evidence/card crops

Detector-only (base.txt deps; NO OCR). Card price/name/barcode +
est-lost-revenue are filled at integration time from the real recognition
pipeline. Never touches the graded 29-column CSV / submission path.
See docs/shelf-audit.md (P2/P3).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections import defaultdict
from pathlib import Path

import cv2

from price_tag_pipeline.config import DetectorConfig
from price_tag_pipeline.detector import build_detector, read_video_fps
from price_tag_pipeline.shelf_analytics import (
    AssociationConfig,
    CardConfig,
    ShelfAlert,
    ShelfAudit,
    TrackTrace,
    associate_tracks,
    build_card_set,
    regroup_missing_price_tag,
)
from price_tag_pipeline.shelf_analytics.schema import AlertType

MAIN_TREE = Path("E:/Hackatons/lenta_tech_life_2026")
DEFAULT_PRODUCT_W = MAIN_TREE / "data/checkpoints/product_detector/product_facing_retail_pretrain_yolo11s_best.pt"
DEFAULT_TAG_W = MAIN_TREE / "data/checkpoints/detector/best.pt"
VIDEO_DIR = MAIN_TREE / "data/raw/videos"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--product-weights", type=Path, default=DEFAULT_PRODUCT_W)
    p.add_argument("--tag-weights", type=Path, default=DEFAULT_TAG_W)
    p.add_argument("--product-conf", type=float, default=0.30)
    p.add_argument("--tag-conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--rotation", default="ccw")
    p.add_argument("--max-frames", type=int, default=0, help="0 = whole video")
    p.add_argument("--persistence", type=int, default=10)
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


def _sharpness(crop) -> float:
    if crop is None or crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return 0.0
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def _collect_traces(cfg: DetectorConfig, video: Path, max_frames: int,
                    capture_best: bool = False):
    """Stream once → traces + (W,H) [+ best {tid:(sharpness,frame,bbox)}]."""
    det = build_detector(cfg)
    boxes: dict[int, dict[int, tuple]] = defaultdict(dict)
    best: dict[int, tuple] = {}
    frame_w = frame_h = 0
    for frame_idx, (frame, dets) in enumerate(det.stream_video(str(video))):
        if frame_idx == 0:
            frame_h, frame_w = frame.shape[:2]
        for d in dets:
            if d.track_id is None:
                continue
            bb = tuple(float(v) for v in d.bbox_xyxy)
            boxes[d.track_id][frame_idx] = bb
            if capture_best:
                x1, y1, x2, y2 = (int(v) for v in d.bbox_xyxy)
                if x2 - x1 >= 10 and y2 - y1 >= 10:
                    s = _sharpness(frame[y1:y2, x1:x2])
                    if d.track_id not in best or s > best[d.track_id][0]:
                        best[d.track_id] = (s, frame_idx, bb)
        if max_frames and frame_idx + 1 >= max_frames:
            break
    traces = [TrackTrace(track_id=tid, boxes=bx) for tid, bx in boxes.items()]
    return traces, frame_w, frame_h, best


def _grab_frames(video: Path, wanted: set[int]) -> dict[int, "cv2.Mat"]:
    if not wanted:
        return {}
    cap = cv2.VideoCapture(str(video))
    out: dict[int, "cv2.Mat"] = {}
    last, idx = max(wanted), -1
    while idx < last:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx in wanted:
            out[idx] = frame
    cap.release()
    return out


def _crop(frame, bbox, pad: float = 0.06):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    x1 = int(max(0, x1 - bw * pad)); y1 = int(max(0, y1 - bh * pad))
    x2 = int(min(w, x2 + bw * pad)); y2 = int(min(h, y2 + bh * pad))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return frame[y1:y2, x1:x2]


def _hist(crop) -> tuple:
    """Normalised HS colour histogram → appearance vector for SKU similarity."""
    if crop is None or crop.size == 0:
        return ()
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
    cv2.normalize(h, h)
    return tuple(float(v) for v in h.flatten())


def audit_video(video: Path, args: argparse.Namespace) -> dict:
    print(f"\n=== {video.name} ===")
    fps = read_video_fps(str(video))

    print("[1/4] product-facing detector ...")
    prod_traces, fw, fh, best = _collect_traces(
        _cfg(args.product_weights, args.product_conf, args.imgsz, args.rotation, args.device),
        video, args.max_frames, capture_best=True)
    print(f"      {len(prod_traces)} product tracks  (frame {fw}x{fh})")

    print("[2/4] price-tag detector ...")
    tag_traces, fw2, fh2, _ = _collect_traces(
        _cfg(args.tag_weights, args.tag_conf, args.imgsz, args.rotation, args.device),
        video, args.max_frames)
    fw, fh = fw or fw2, fh or fh2
    print(f"      {len(tag_traces)} price-tag tracks")

    print("[3/4] associate ...")
    res = associate_tracks(
        tag_traces, prod_traces, frame_w=fw, frame_h=fh, fps=fps,
        cfg=AssociationConfig(rotation=args.rotation,
                              persistence_min_frames=args.persistence))

    out_dir = args.outdir / video.stem
    crops_dir = out_dir / "crops"
    cards_dir = out_dir / "cards"
    crops_dir.mkdir(parents=True, exist_ok=True)
    cards_dir.mkdir(parents=True, exist_ok=True)

    tag_by_id = {t.track_id: t for t in tag_traces}

    # --- frames needed for appearance + card images + evidence crops --------
    need = {f for s, f, b in best.values()}
    persistent_oos = [u for u in res.unmatched_price_tags if u.persistent]
    for u in persistent_oos:
        fr = sorted(tag_by_id[u.track_id].boxes)
        need.add(fr[len(fr) // 2])
    frames = _grab_frames(video, need)

    appearance: dict[int, tuple] = {}
    for tid, (s, f, b) in best.items():
        fr = frames.get(f)
        if fr is not None:
            appearance[tid] = _hist(_crop(fr, b))

    print("[4/4] cards + group-level alerts ...")
    card_set = build_card_set(
        prod_traces, frame_w=fw, frame_h=fh, fps=fps, video_id=video.stem,
        appearance=appearance or None,
        best_track=lambda tid: best.get(tid, (0.0,))[0],
        cfg=CardConfig(rotation=args.rotation))

    # Save one best-crop image per card; fill ProductCard.best_crop.
    cards = []
    for c in card_set.cards:
        rel = None
        b = best.get(c.product_track_id)
        if b is not None and (fr := frames.get(b[1])) is not None:
            crop = _crop(fr, b[2])
            if crop is not None and cv2.imwrite(str(cards_dir / f"{c.card_id}.jpg"), crop):
                rel = f"cards/{c.card_id}.jpg"
        cards.append(dataclasses.replace(c, best_crop=rel))
    card_by_id = {c.card_id: c for c in cards}

    # --- alerts -----------------------------------------------------------
    alerts: list[ShelfAlert] = []
    for u in persistent_oos:
        fr_list = sorted(tag_by_id[u.track_id].boxes)
        fi = fr_list[len(fr_list) // 2]
        bbox = tag_by_id[u.track_id].boxes.get(fi, u.representative_bbox)
        aid = f"oos_{video.stem}_{u.track_id:04d}"
        rel = None
        if (fr := frames.get(fi)) is not None and (cp := _crop(fr, bbox)) is not None:
            if cv2.imwrite(str(crops_dir / f"{aid}.jpg"), cp):
                rel = f"crops/{aid}.jpg"
        alerts.append(ShelfAlert(
            id=aid, type=AlertType.OUT_OF_STOCK, severity="high",
            video_id=video.stem, timestamp_s=fi / (fps if fps > 1e-6 else 1.0),
            frame_idx=fi, bbox_xyxy=bbox, track_id=u.track_id,
            evidence_crop=rel, first_seen_s=u.first_s, last_seen_s=u.last_s,
            persistence_frames=u.n_frames))

    # MISSING_PRICE_TAG is now per *card* (group), with a group persistence gate.
    member_frames = {
        c.card_id: len({f for tid in card_set.card_members[c.card_id]
                        for t in prod_traces if t.track_id == tid for f in t.boxes})
        for c in cards
    }
    missing_cards = regroup_missing_price_tag(res, card_set)
    filtered_cards = 0
    for cid in missing_cards:
        c = card_by_id[cid]
        if member_frames[cid] < args.persistence:
            filtered_cards += 1
            continue
        aid = f"notag_{video.stem}_{cid.split('_')[-1]}"
        rb = best.get(c.product_track_id)  # (sharpness, frame_idx, bbox)
        a_fi = rb[1] if rb else -1
        a_bbox = rb[2] if rb else (0.0, 0.0, 0.0, 0.0)
        a_ts = (a_fi / fps) if (rb and fps > 1e-6) else (c.seen_from_s or 0.0)
        alerts.append(ShelfAlert(
            id=aid, type=AlertType.MISSING_PRICE_TAG, severity="high",
            video_id=video.stem,
            timestamp_s=a_ts, frame_idx=a_fi,
            bbox_xyxy=a_bbox, track_id=c.product_track_id,
            evidence_crop=c.best_crop,
            product={"card_id": cid, "facing_count": c.facing_count,
                     "member_track_ids": list(card_set.card_members[cid])},
            first_seen_s=c.seen_from_s, last_seen_s=c.seen_to_s,
            persistence_frames=member_frames[cid]))

    n_oos = sum(a.type is AlertType.OUT_OF_STOCK for a in alerts)
    n_notag = sum(a.type is AlertType.MISSING_PRICE_TAG for a in alerts)
    n_ok = sum(r.status == "ok" for r in res.relations)
    summary = {
        "n_out_of_stock": n_oos,
        "n_missing_price_tag": n_notag,
        "n_product_cards": len(cards),
        "n_relations_ok": n_ok,
        "n_relations_ambiguous": len(res.relations) - n_ok,
        "n_product_tracks": len(prod_traces),
        "n_price_tag_tracks": len(tag_traces),
        "missing_tag_facing_level_would_be": len(
            [t for t in prod_traces if t.track_id not in {
                int(r.product_group_id.split("_")[-1])
                for r in res.relations
                if r.status == "ok" and r.product_group_id}]),
        "filtered_non_persistent_tags": sum(
            not u.persistent for u in res.unmatched_price_tags),
        "filtered_non_persistent_cards": filtered_cards,
        "est_lost_revenue_rub": None,  # integration-time (real OCR)
    }
    audit = ShelfAudit(
        video_id=video.stem, fps=fps, summary=summary,
        relations=res.relations, alerts=tuple(alerts), cards=tuple(cards),
        metadata={
            "product_weights": str(args.product_weights),
            "tag_weights": str(args.tag_weights),
            "product_conf": args.product_conf, "tag_conf": args.tag_conf,
            "rotation": args.rotation, "max_frames": args.max_frames,
            "persistence_min_frames": args.persistence,
            "note": "detector-only; CSV/submission path untouched; "
                    "card price/name/barcode filled at integration (real OCR)",
        })
    (out_dir / "audit.json").write_text(
        json.dumps(audit.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "alerts.json").write_text(
        json.dumps([a.to_dict() for a in alerts], ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"      {len(alerts)} alerts, {len(cards)} cards -> {out_dir}")
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
