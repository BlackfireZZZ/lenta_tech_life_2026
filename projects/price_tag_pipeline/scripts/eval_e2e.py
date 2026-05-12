#!/usr/bin/env python3
"""End-to-end evaluation: predicted JSONL vs ground-truth JSONL.

Matching strategy:
    1. Group by video_id.
    2. Greedily match predicted track to ground-truth tag by bbox IoU (≥0.3).
    3. Each unmatched ground truth counts as a miss (E2E = 0 for that tag).
    4. Each unmatched prediction counts as a false positive but is logged separately.

Usage:
    python projects/price_tag_pipeline/scripts/eval_e2e.py \\
        --pred outputs/video01.jsonl \\
        --gt   data/processed/gt_e2e/video01.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

THIS = Path(__file__).resolve().parent
SRC = THIS.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from price_tag_pipeline.metrics.e2e import e2e_field_accuracy, per_field_report  # noqa: E402


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / max(1, area_a + area_b - inter)


def _greedy_match(preds, gts, iou_thresh: float = 0.3):
    pairs: list[tuple[dict, dict]] = []
    used_gt = set()
    # Sort preds by overall_confidence descending.
    preds = sorted(preds, key=lambda x: -x.get("overall_confidence", 0.0))
    for p in preds:
        pbox = p["bbox"]
        best_iou = 0.0
        best_idx = -1
        for j, g in enumerate(gts):
            if j in used_gt:
                continue
            gb = g["bbox"]
            i = _iou(pbox, gb)
            if i > best_iou:
                best_iou = i
                best_idx = j
        if best_idx >= 0 and best_iou >= iou_thresh:
            pairs.append((p, gts[best_idx]))
            used_gt.add(best_idx)
    unmatched_preds = [p for p in preds if not any(p is pp for pp, _ in pairs)]
    unmatched_gts = [g for j, g in enumerate(gts) if j not in used_gt]
    return pairs, unmatched_preds, unmatched_gts


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True)
    p.add_argument("--gt", required=True)
    p.add_argument("--iou-thresh", type=float, default=0.3)
    p.add_argument("--name-tau", type=float, default=0.15)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO)
    preds = _read_jsonl(Path(args.pred))
    gts = _read_jsonl(Path(args.gt))

    pairs, fp, fn = _greedy_match(preds, gts, iou_thresh=args.iou_thresh)
    e2e = e2e_field_accuracy(pairs, name_tau=args.name_tau)
    fields = per_field_report(pairs, name_tau=args.name_tau)

    print(f"\nPaired tags:        {len(pairs)}")
    print(f"False positives:    {len(fp)}")
    print(f"Missed (FN):        {len(fn)}")
    if pairs:
        print(f"\nE2E field accuracy: {e2e:.4f}")
        print("Per-field accuracy:")
        for k, v in fields.items():
            print(f"  {k:30s} {v:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
