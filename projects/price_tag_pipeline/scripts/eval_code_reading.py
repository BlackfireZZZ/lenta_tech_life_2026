#!/usr/bin/env python3
"""Benchmark: does the tracker amplify CODE reading? GT-backed barcode recall.

Controlled A/B of the tracker-amplification lever ONLY: one
detector+tracker+rectify pass per video (identical tracks/crops), then decode
the SAME per-track best-crop buffer with the codes-only chain at two budgets —
``N=top_k`` (old: codes only on OCR's top-K) vs ``N=code_decode_top_k`` (new:
the tracker sweeps cheap decoders over many more of the tag's frames). Only N
differs, so any recall delta is the tracker feeding more frames.

Recall = |decoded GTIN ∩ GT GTIN| / |GT GTIN| per video (GT = barcode /
qr_code_barcode columns of the video's annotation CSV). pyzbar/cv2 only
locally (zxingcpp/pylibdmtx missing) → absolute recall is a floor; the
*relative lift* from more frames is the faithful signal.

    .venv/Scripts/python.exe scripts/eval_code_reading.py --device 0 \\
        --weights E:/.../data/checkpoints/detector/best.pt
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import logging
import re
import sys
import time
from pathlib import Path

THIS = Path(__file__).resolve().parent
SRC = THIS.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

_MAIN = THIS.parents[5]
DEF_VID = _MAIN / "data" / "raw" / "videos"
DEF_GT = _MAIN / "data" / "raw" / "annotations" / "csv"
LOGGER = logging.getLogger("eval_code")

_GTIN_LENS = {8, 12, 13, 14}


def _norm_gtin(v: object) -> str | None:
    d = re.sub(r"\D", "", str(v))
    return d if len(d) in _GTIN_LENS else None


def _gt_gtins(gt_csv: Path) -> set[str]:
    out: set[str] = set()
    with gt_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            for col in ("barcode", "qr_code_barcode"):
                g = _norm_gtin(row.get(col, ""))
                if g:
                    out.add(g)
    return out


def _decoded_gtins(parsed) -> set[str]:
    out: set[str] = set()
    for key in ("barcode", "qr_code_barcode"):
        g = _norm_gtin(parsed.extra_fields.get(key, ""))
        if g:
            out.add(g)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/balanced.yaml")
    p.add_argument("--videos-dir", default=str(DEF_VID))
    p.add_argument("--gt-dir", default=str(DEF_GT))
    p.add_argument("--videos", nargs="*", default=None)
    p.add_argument("--weights", default=None)
    p.add_argument("--device", default="0")
    p.add_argument("--max-frames", type=int, default=4000)
    p.add_argument("--fuse", type=int, default=0,
                   help="also median-fuse this many sharpest crops/track and decode (Level-2)")
    p.add_argument("--json-out", default="runs/code_reading.json")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from price_tag_pipeline.aggregator import TrackAggregator
    from price_tag_pipeline.config import load_config
    from price_tag_pipeline.detector import build_detector
    from price_tag_pipeline.fusion import fuse_crops
    from price_tag_pipeline.recognition import build_recognition_chain
    from price_tag_pipeline.rectifier import build_rectifier

    cfg = load_config(args.config)
    det_cfg = dataclasses.replace(cfg.detector, device=args.device,
                                  **({"model_path": args.weights} if args.weights else {}))
    code_chain = build_recognition_chain(
        dataclasses.replace(cfg, recognition=dataclasses.replace(
            cfg.recognition, enable_ocr=False)))
    top_k = cfg.ocr.top_k_crops_per_track
    wide_k = cfg.ocr.code_decode_top_k or (1 << 30)
    LOGGER.info("code chain=%s  N: old=%d new=%d  rotate=%s  weights=%s",
                code_chain.decoder_names, top_k, wide_k,
                cfg.detector.frame_rotation, det_cfg.model_path)

    vdir, gdir = Path(args.videos_dir), Path(args.gt_dir)
    stems = args.videos or sorted(
        v.stem for v in vdir.glob("*.mp4") if (gdir / f"{v.stem}.csv").is_file())

    rect = build_rectifier(cfg.rectifier)
    totals = {"gt": 0, "old_hit": 0, "new_hit": 0, "fused_hit": 0,
              "old_dec": 0, "new_dec": 0}
    reports = []
    for stem in stems:
        gt = _gt_gtins(gdir / f"{stem}.csv")
        det = build_detector(det_cfg)
        agg = TrackAggregator(cfg.aggregation)
        n = 0
        t0 = time.time()
        for frame, dets in det.stream_video(str(vdir / f"{stem}.mp4")):
            fh, fw = frame.shape[:2]
            for d in dets:
                if d.track_id is None or d.confidence < cfg.ocr.min_detection_confidence:
                    continue
                crop = rect.rectify(frame, d)
                if crop is None or crop.sharpness < cfg.ocr.min_sharpness \
                        or crop.area_px < cfg.ocr.min_crop_area_px:
                    continue
                agg.push_crop(d.track_id, crop, frame_w=fw, frame_h=fh)
            n += 1
            if n >= args.max_frames:
                break

        old_dec: set[str] = set()
        new_dec: set[str] = set()
        fused_dec: set[str] = set()
        for tid in list(agg._tracks.keys()):
            entries = agg.best_crops(tid, wide_k)
            for i, e in enumerate(entries):
                g: set[str] = set()
                for r in code_chain.decode(e.crop.image):
                    g |= _decoded_gtins(r.parsed)
                if g:
                    new_dec |= g
                    if i < top_k:
                        old_dec |= g
            if args.fuse > 0 and len(entries) >= 2:
                fimg = fuse_crops([e.crop.image for e in entries[:args.fuse]],
                                  max_frames=args.fuse)
                if fimg is not None:
                    for r in code_chain.decode(fimg):
                        fused_dec |= _decoded_gtins(r.parsed)
        # Level-2 is additive to the per-frame decodes (it rescues tags no
        # single frame got): union with the wide-budget set.
        fused_all = new_dec | fused_dec
        oh, nh = len(old_dec & gt), len(new_dec & gt)
        fh = len(fused_all & gt)
        reports.append({"video": stem, "gt": len(gt),
                        "old_recall": round(oh / max(1, len(gt)), 3),
                        "new_recall": round(nh / max(1, len(gt)), 3),
                        "fused_recall": round(fh / max(1, len(gt)), 3),
                        "old_hit": oh, "new_hit": nh, "fused_hit": fh,
                        "old_dec": len(old_dec), "new_dec": len(new_dec)})
        totals["gt"] += len(gt)
        totals["old_hit"] += oh
        totals["new_hit"] += nh
        totals["fused_hit"] += fh
        totals["old_dec"] += len(old_dec)
        totals["new_dec"] += len(new_dec)
        LOGGER.info(
            "%-12s gt=%-3d  recall old(N=%d)=%.2f new(N=%d)=%.2f "
            "fused(+%d)=%.2f  hits %d/%d/%d  (%.0fs)",
            stem, len(gt), top_k, reports[-1]["old_recall"], wide_k,
            reports[-1]["new_recall"], args.fuse, reports[-1]["fused_recall"],
            oh, nh, fh, time.time() - t0)

    g = max(1, totals["gt"])
    summ = {"videos": len(reports), "gt_total": totals["gt"],
            "old_recall": round(totals["old_hit"] / g, 3),
            "new_recall": round(totals["new_hit"] / g, 3),
            "fused_recall": round(totals["fused_hit"] / g, 3),
            "old_hits": totals["old_hit"], "new_hits": totals["new_hit"],
            "fused_hits": totals["fused_hit"]}
    LOGGER.info("SUMMARY %s", json.dumps(summ, ensure_ascii=False))
    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summ, "reports": reports},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
