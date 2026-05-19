#!/usr/bin/env python3
"""P0 reality check for the shelf-audit killer feature.

Runs the friend's pretrained `product_facing` detector AND the production
price-tag detector over a real Lenta video, on the SAME rotation-corrected
stream (`build_detector` / `stream_video`, rotate=ccw), and:

  * dumps annotated sample frames (product=green, price-tag=magenta) so we can
    eyeball orientation and box quality;
  * prints detection-count and confidence stats to calibrate `conf`;
  * runs a geometry sanity check on the "price tag sits BELOW its product"
    assumption (the matcher relies on it).

This is a *probe*, not pipeline code: read-only, writes only under --outdir,
never touches the graded CSV path. See docs/shelf-audit.md (P0).
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import cv2

from price_tag_pipeline.config import DetectorConfig
from price_tag_pipeline.detector import build_detector
from price_tag_pipeline.types import Detection

# Checkpoints are .gitignore'd, so they live only in the main working tree.
MAIN_TREE = Path("E:/Hackatons/lenta_tech_life_2026")
DEFAULT_PRODUCT_W = MAIN_TREE / "data/checkpoints/product_detector/product_facing_retail_pretrain_yolo11s_best.pt"
DEFAULT_TAG_W = MAIN_TREE / "data/checkpoints/detector/best.pt"
DEFAULT_VIDEO = MAIN_TREE / "data/raw/videos/25_12-20.mp4"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    p.add_argument("--product-weights", type=Path, default=DEFAULT_PRODUCT_W)
    p.add_argument("--tag-weights", type=Path, default=DEFAULT_TAG_W)
    p.add_argument("--product-conf", type=float, default=0.25)
    p.add_argument("--tag-conf", type=float, default=0.25)
    p.add_argument("--product-imgsz", type=int, default=1280)
    p.add_argument("--tag-imgsz", type=int, default=1280)
    p.add_argument("--max-frames", type=int, default=60,
                   help="process only the first N frames (CPU is slow)")
    p.add_argument("--every", type=int, default=10,
                   help="dump an annotated frame every N processed frames")
    p.add_argument("--device", default=None, help="ultralytics device (None=auto/CPU)")
    p.add_argument("--outdir", type=Path, default=None)
    return p.parse_args()


def _cfg(model_path: Path, conf: float, imgsz: int, device) -> DetectorConfig:
    return DetectorConfig(
        backend="yolo",
        model_path=str(model_path),
        conf=conf,
        iou=0.50,
        device=device,
        classes=None,
        tracker_yaml="configs/trackers/bytetrack.yaml",
        image_size=imgsz,
        frame_rotation="ccw",  # robot cam is 90° CW; models trained upright
    )


def _collect(cfg: DetectorConfig, video: Path, max_frames: int):
    """Return {frame_idx: (frame_bgr, [Detection])} for the first N frames."""
    det = build_detector(cfg)
    out: dict[int, tuple] = {}
    for frame_idx, (frame, dets) in enumerate(det.stream_video(str(video))):
        out[frame_idx] = (frame, list(dets))
        if frame_idx + 1 >= max_frames:
            break
    return out


def _stats(by_frame: dict[int, tuple]) -> dict:
    confs: list[float] = []
    per_frame: list[int] = []
    for _frame, dets in by_frame.values():
        per_frame.append(len(dets))
        confs.extend(d.confidence for d in dets)
    if not confs:
        return {"total": 0, "frames": len(by_frame), "per_frame_mean": 0.0,
                "conf": None}
    confs.sort()

    def pct(q: float) -> float:
        return round(confs[min(len(confs) - 1, int(q * len(confs)))], 4)

    return {
        "total": len(confs),
        "frames": len(by_frame),
        "per_frame_mean": round(statistics.mean(per_frame), 2),
        "per_frame_max": max(per_frame),
        "conf": {"min": round(confs[0], 4), "p25": pct(0.25),
                 "median": pct(0.5), "p75": pct(0.75), "max": round(confs[-1], 4)},
    }


def _to_upright(box, orig_w: int) -> tuple[float, float, float, float]:
    """Forward map an ORIGINAL-frame box into the CCW-upright space.

    Inverse of detector.unrotate_box_xyxy(..., 'ccw'): that maps rotated->orig
    as ox = orig_w - y, oy = x. Inverting: x_up = oy, y_up = orig_w - ox.
    """
    ox1, oy1, ox2, oy2 = box
    xs = sorted((oy1, oy2))
    ys = sorted((orig_w - ox1, orig_w - ox2))
    return xs[0], ys[0], xs[1], ys[1]


def _tag_below_product(tags: list[Detection], prods: list[Detection],
                        orig_w: int | None = None) -> tuple[int, int]:
    """Fraction of tags that have a product overlapping in x and ABOVE them.

    With ``orig_w`` set, both boxes are first mapped into CCW-upright space
    (the space the matcher's "tag below product" assumption is valid in);
    without it the raw original-frame space is used (expected to be poor).
    """
    ok = 0
    for t in tags:
        tb = _to_upright(t.bbox_xyxy, orig_w) if orig_w else tuple(map(float, t.bbox_xyxy))
        tx1, ty1, tx2, ty2 = tb
        t_cy = (ty1 + ty2) / 2.0
        for pr in prods:
            pb = _to_upright(pr.bbox_xyxy, orig_w) if orig_w else tuple(map(float, pr.bbox_xyxy))
            px1, _py1, px2, py2 = pb
            x_overlap = min(tx2, px2) - max(tx1, px1)
            if x_overlap > 0 and py2 <= t_cy:  # product bottom above tag centre
                ok += 1
                break
    return ok, len(tags)


def _draw(frame, dets: list[Detection], color, tag: str):
    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d.bbox_xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{tag}{d.track_id if d.track_id is not None else '?'} {d.confidence:.2f}"
        cv2.putText(frame, label, (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def main() -> int:
    args = parse_args()
    for w in (args.product_weights, args.tag_weights):
        if not w.exists():
            raise SystemExit(f"Missing checkpoint: {w}")
    if not args.video.exists():
        raise SystemExit(f"Missing video: {args.video}")

    outdir = args.outdir or (MAIN_TREE / "outputs/shelf_audit/_p0" / args.video.stem)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[p0] video={args.video.name}  outdir={outdir}")

    print("[p0] running product_facing detector ...")
    prod_by_frame = _collect(
        _cfg(args.product_weights, args.product_conf, args.product_imgsz, args.device),
        args.video, args.max_frames)
    print("[p0] running price-tag detector ...")
    tag_by_frame = _collect(
        _cfg(args.tag_weights, args.tag_conf, args.tag_imgsz, args.device),
        args.video, args.max_frames)

    prod_stats = _stats(prod_by_frame)
    tag_stats = _stats(tag_by_frame)

    orig_w = next(iter(prod_by_frame.values()))[0].shape[1]
    below_ok = below_tot = 0          # original (sideways) space
    up_ok = up_tot = 0                # CCW-upright space
    dumped = []
    for fi in sorted(prod_by_frame):
        prods = prod_by_frame[fi][1]
        tags = tag_by_frame.get(fi, (None, []))[1]
        o, n = _tag_below_product(tags, prods)
        below_ok += o
        below_tot += n
        uo, un = _tag_below_product(tags, prods, orig_w=orig_w)
        up_ok += uo
        up_tot += un
        if fi % args.every == 0:
            frame = prod_by_frame[fi][0].copy()
            _draw(frame, prods, (0, 200, 0), "P")
            _draw(frame, tags, (200, 0, 200), "T")
            path = outdir / f"frame_{fi:05d}.jpg"
            cv2.imwrite(str(path), frame)
            dumped.append(path.name)

    summary = {
        "video": str(args.video),
        "product": {"weights": str(args.product_weights),
                    "conf": args.product_conf, "imgsz": args.product_imgsz,
                    "stats": prod_stats},
        "price_tag": {"weights": str(args.tag_weights),
                      "conf": args.tag_conf, "imgsz": args.tag_imgsz,
                      "stats": tag_stats},
        "tag_below_product_original_space": {
            "ok": below_ok, "total": below_tot,
            "ratio": round(below_ok / below_tot, 3) if below_tot else None,
        },
        "tag_below_product_upright_space": {
            "ok": up_ok, "total": up_tot,
            "ratio": round(up_ok / up_tot, 3) if up_tot else None,
            "note": "associator MUST work here: forward-rotate boxes to CCW-upright, then matcher geometry holds",
        },
        "dumped_frames": dumped,
    }
    (outdir / "p0_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[p0] wrote {len(dumped)} annotated frames + p0_summary.json -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
