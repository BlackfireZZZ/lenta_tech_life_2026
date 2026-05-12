#!/usr/bin/env python3
"""Evaluate predictions in hackathon CSV format against ground truth CSV.

Primary score:
  share of GT tags whose per-tag field accuracy is >= 0.80

Matching:
  - by `filename`
  - then greedy max-IoU matching on bbox (when bbox exists)
  - optional timestamp fallback for rows without bbox
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


GEOM_FIELDS = {"filename", "frame_timestamp", "x_min", "y_min", "x_max", "y_max"}
DEFAULT_GRADED_FIELDS = [
    "product_name",
    "price_default",
    "price_card",
    "price_discount",
    "barcode",
    "discount_amount",
    "id_sku",
    "print_datetime",
    "code",
    "additional_info",
    "color",
    "special_symbols",
    "qr_code_barcode",
    "price1_qr",
    "price2_qr",
    "price3_qr",
    "price4_qr",
    "wholesale_level_1_count",
    "wholesale_level_1_price",
    "wholesale_level_2_count",
    "wholesale_level_2_price",
    "action_price_qr",
    "action_code_qr",
]

NUMERIC_FIELDS = {
    "price_default",
    "price_card",
    "price_discount",
    "discount_amount",
    "price1_qr",
    "price2_qr",
    "price3_qr",
    "price4_qr",
    "wholesale_level_1_count",
    "wholesale_level_1_price",
    "wholesale_level_2_count",
    "wholesale_level_2_price",
    "action_price_qr",
}


@dataclass(frozen=True)
class MatchPair:
    pred: dict[str, str]
    gt: dict[str, str]


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


def _to_float(x: str) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip().replace(",", ".")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _to_int(x: str) -> Optional[int]:
    xf = _to_float(x)
    if xf is None:
        return None
    return int(round(xf))


def _bbox(row: dict[str, str]) -> Optional[tuple[int, int, int, int]]:
    x1 = _to_int(row.get("x_min", ""))
    y1 = _to_int(row.get("y_min", ""))
    x2 = _to_int(row.get("x_max", ""))
    y2 = _to_int(row.get("y_max", ""))
    if None in (x1, y1, x2, y2):
        return None
    return (x1, y1, x2, y2)


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
    denom = max(1, area_a + area_b - inter)
    return inter / denom


def _norm_text(x: str) -> str:
    return " ".join(str(x or "").strip().lower().split())


def _field_match(field: str, pred_v: str, gt_v: str, numeric_tol: float) -> bool:
    gt_n = _norm_text(gt_v)
    pred_n = _norm_text(pred_v)

    # Empty GT = unknown, skip at caller level.
    if not gt_n:
        return True

    if field in NUMERIC_FIELDS:
        g = _to_float(gt_n)
        p = _to_float(pred_n)
        if g is None:
            return pred_n == gt_n
        if p is None:
            return False
        return abs(p - g) <= numeric_tol

    return pred_n == gt_n


def _per_tag_accuracy(
    pred: dict[str, str],
    gt: dict[str, str],
    fields: list[str],
    numeric_tol: float,
) -> float:
    considered = 0
    correct = 0
    for f in fields:
        gt_v = gt.get(f, "")
        if not _norm_text(gt_v):
            continue
        considered += 1
        if _field_match(f, pred.get(f, ""), gt_v, numeric_tol=numeric_tol):
            correct += 1
    if considered == 0:
        return 0.0
    return correct / considered


def _group_by_filename(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        fn = r.get("filename", "")
        out.setdefault(fn, []).append(r)
    return out


def _greedy_match(
    pred_rows: list[dict[str, str]],
    gt_rows: list[dict[str, str]],
    iou_thresh: float,
    ts_window_ms: int,
) -> tuple[list[MatchPair], int, int]:
    matches: list[MatchPair] = []
    used_pred: set[int] = set()
    used_gt: set[int] = set()

    for gi, gt in enumerate(gt_rows):
        gb = _bbox(gt)
        gts = _to_int(gt.get("frame_timestamp", ""))
        best_idx = -1
        best_score = -1.0
        for pi, pred in enumerate(pred_rows):
            if pi in used_pred:
                continue
            pb = _bbox(pred)
            if gb is not None and pb is not None:
                score = _iou(gb, pb)
                if score < iou_thresh:
                    continue
            else:
                pts = _to_int(pred.get("frame_timestamp", ""))
                if gts is None or pts is None:
                    continue
                if abs(pts - gts) > ts_window_ms:
                    continue
                # Timestamp fallback: closer is better.
                score = 1.0 - min(1.0, abs(pts - gts) / max(1.0, float(ts_window_ms)))
            if score > best_score:
                best_score = score
                best_idx = pi

        if best_idx >= 0:
            used_pred.add(best_idx)
            used_gt.add(gi)
            matches.append(MatchPair(pred=pred_rows[best_idx], gt=gt))

    fp = len(pred_rows) - len(used_pred)
    fn = len(gt_rows) - len(used_gt)
    return matches, fp, fn


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pred-csv", required=True, help="Predicted CSV path")
    p.add_argument("--gt-csv", required=True, help="Ground-truth CSV path")
    p.add_argument("--iou-thresh", type=float, default=0.3)
    p.add_argument("--tag-pass-threshold", type=float, default=0.80)
    p.add_argument("--numeric-tol", type=float, default=0.01)
    p.add_argument("--ts-window-ms", type=int, default=1200)
    args = p.parse_args()

    pred_rows = _load_csv(Path(args.pred_csv).expanduser().resolve())
    gt_rows = _load_csv(Path(args.gt_csv).expanduser().resolve())
    pred_by_fn = _group_by_filename(pred_rows)
    gt_by_fn = _group_by_filename(gt_rows)

    all_matches: list[MatchPair] = []
    total_fp = 0
    total_fn = 0
    for filename, gt_chunk in gt_by_fn.items():
        pred_chunk = pred_by_fn.get(filename, [])
        matched, fp, fn = _greedy_match(
            pred_chunk,
            gt_chunk,
            iou_thresh=args.iou_thresh,
            ts_window_ms=args.ts_window_ms,
        )
        all_matches.extend(matched)
        total_fp += fp
        total_fn += fn

    per_tag_scores = [
        _per_tag_accuracy(
            m.pred,
            m.gt,
            fields=DEFAULT_GRADED_FIELDS,
            numeric_tol=args.numeric_tol,
        )
        for m in all_matches
    ]
    passed = sum(1 for s in per_tag_scores if s >= args.tag_pass_threshold)
    matched_total = len(per_tag_scores)
    gt_total = len(gt_rows)
    pass_share = (passed / gt_total) if gt_total > 0 else 0.0
    mean_acc = (sum(per_tag_scores) / matched_total) if matched_total > 0 else 0.0

    print(f"GT tags:                     {gt_total}")
    print(f"Matched tags:                {matched_total}")
    print(f"False positives (unmatched): {total_fp}")
    print(f"Missed GT (FN):              {total_fn}")
    print(f"Mean matched tag accuracy:   {mean_acc:.4f}")
    print(
        f"Share of GT tags with accuracy >= {args.tag_pass_threshold:.2f}: "
        f"{pass_share:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
