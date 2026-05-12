#!/usr/bin/env python3
"""Evaluate a trained YOLO-family detector on a fold's val set.

Reads a fold manifest, builds a temporary dataset.yaml pointing at the val
videos, runs Ultralytics' built-in validator, and prints the metric block.

Usage:
    python projects/price_tag_pipeline/scripts/eval_detector.py \\
        --weights runs/lenta/yolo26l_fold0/weights/best.pt \\
        --dataset data/processed/dataset.yaml \\
        --imgsz 1280 --device 0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
SRC = THIS.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="0")
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.6)
    p.add_argument("--save-json", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO)
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit("ultralytics not installed: pip install ultralytics") from e

    model = YOLO(args.weights)
    results = model.val(
        data=args.dataset,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        save_json=args.save_json,
        verbose=True,
    )
    box = results.box
    print(
        f"\nmAP@0.5:       {box.map50:.4f}\n"
        f"mAP@0.5:0.95:  {box.map:.4f}\n"
        f"mP:            {box.mp:.4f}\n"
        f"mR:            {box.mr:.4f}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
