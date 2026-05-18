"""Diagnostic: are organizer gt_e2e boxes usable? Dump crop image + raw VLM
output + GT row side by side so we can SEE box-noise vs parser/format issues.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent / "src"))

from price_tag_pipeline import cv_io
from price_tag_pipeline.config import load_config
from price_tag_pipeline.recognition.ocr import build_ocr_engine
from price_tag_pipeline.rectifier import build_rectifier
from price_tag_pipeline.types import Detection

ap = argparse.ArgumentParser()
ap.add_argument("--video", default="49_5")
ap.add_argument("--n", type=int, default=4)
ap.add_argument("--backend", default="glm_ocr")
ap.add_argument("--vlm-model", default=str(THIS.parents[3] / "weights" / "glm-ocr"))
ap.add_argument("--config", default=str(THIS.parent / "configs" / "balanced.yaml"))
ap.add_argument("--out", default=str(THIS.parents[3] / "outputs" / "diag"))
args = ap.parse_args()

cfg = load_config(args.config)
cfg = dataclasses.replace(cfg, ocr=dataclasses.replace(
    cfg.ocr, backend=args.backend, vlm_model=args.vlm_model))
rect = build_rectifier(cfg.rectifier)
eng = build_ocr_engine(cfg.ocr)

gt_root = Path("E:/Hackatons/lenta_tech_life_2026/data/processed/gt_e2e")
frames_root = Path("E:/Hackatons/lenta_tech_life_2026/data/processed/frames")
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

rows = [json.loads(l) for l in (gt_root / f"{args.video}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()][: args.n]
for i, row in enumerate(rows):
    fidx = int(row["frame_idx"])
    frame = cv_io.imread(str(frames_root / args.video / f"{fidx:06d}.jpg"))
    if frame is None:
        print(f"[{i}] frame missing"); continue
    h, w = frame.shape[:2]
    b = row["bbox"]
    box = (max(0, int(b[0])), max(0, int(b[1])), min(w - 1, int(b[2])), min(h - 1, int(b[3])))
    det = Detection(frame_idx=fidx, timestamp_s=0.0, bbox_xyxy=box,
                    confidence=1.0, class_id=0, class_name="price_tag", track_id=0)
    cand = rect.rectify(frame, det)
    if cand is None:
        print(f"[{i}] degenerate crop {box}"); continue
    cp = out / f"{args.video}_{i}_{fidx}.jpg"
    cv_io.imwrite(str(cp), cand.image)
    res = eng.recognize(cand.image)
    print(f"\n===== [{i}] {args.video} frame {fidx} box={box} crop={cand.image.shape[1]}x{cand.image.shape[0]} -> {cp}")
    print("GT  :", json.dumps({k: row.get(k) for k in
          ("product_name", "price_default", "price_card", "barcode", "id_sku",
           "print_datetime", "code")}, ensure_ascii=False))
    print("PRED:", res.text[:600].replace("\n", " "))
