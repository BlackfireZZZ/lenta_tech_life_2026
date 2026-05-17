#!/usr/bin/env python3
"""Operating-point eval + visual QA + experiment ledger for the detector.

Why this exists (see docs/runbooks/detector-experiments.md):

- `eval_detector.py` reports Ultralytics' PR-curve mAP at conf~0.001. That is
  the right number for *model capacity* but not for the *shipped operating
  point*. The pipeline runs at a real `conf` (recall-first, ~0.05-0.15), so we
  also need recall/precision **at that conf** with IoU-0.5 matching.
- Every run must be comparable: one row appended to `runs/ledger.csv`.
- The user validates by *looking*: we render GT (red) vs pred (green) frames,
  worst frames first (FN/FP heavy), so the failure modes are obvious.

Usage:
    .venv/Scripts/python.exe projects/price_tag_pipeline/scripts/eval_qa.py \
        --weights runs/lenta/s_builtin_imgsz1280_fold0/weights/best.pt \
        --dataset E:/Hackatons/lenta_tech_life_2026/data/processed/dataset.yaml \
        --imgsz 1280 --conf 0.10 --device 0 --run-id s_builtin_imgsz1280_fold0 \
        --model yolo11s --aug builtin --fold 0 --epochs 200
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
SRC = THIS.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from price_tag_pipeline.cv_io import imread, imwrite  # noqa: E402

LOGGER = logging.getLogger("eval_qa")

LEDGER_COLUMNS = [
    "run_id", "model", "imgsz", "aug", "fold", "epochs", "conf", "iou_match",
    "n_images", "n_gt", "tp", "fp", "fn", "recall", "precision",
    "map50", "map5095", "avg_det_per_frame", "weights_path", "notes",
]


def _read_yaml_min(path: Path) -> dict:
    """Tiny YAML reader for the flat dataset.yaml we emit (no pyyaml dep here)."""
    data: dict = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].rstrip()
        if not line or line.startswith(" ") or ":" not in line:
            continue
        key, _, val = line.partition(":")
        data[key.strip()] = val.strip()
    return data


def _label_path_for(image_path: Path) -> Path:
    """Derive the YOLO label path from an image path.

    Ultralytics convention: swap an `/images/` or `/frames/` path segment for
    `/labels/` and the suffix for `.txt`.
    """
    parts = list(image_path.parts)
    for seg in ("images", "frames"):
        if seg in parts:
            parts[len(parts) - 1 - parts[::-1].index(seg)] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def _read_gt_boxes(label_path: Path, w: int, h: int) -> np.ndarray:
    """Read YOLO-normalized labels -> pixel xyxy array (N,4). Empty -> (0,4)."""
    if not label_path.exists():
        return np.zeros((0, 4), dtype=np.float32)
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cx, cy, bw, bh = (float(p[1]) * w, float(p[2]) * h, float(p[3]) * w, float(p[4]) * h)
        boxes.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
    return np.asarray(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def _match(pred: np.ndarray, pred_conf: np.ndarray, gt: np.ndarray, iou_thr: float):
    """Greedy TP/FP/FN by confidence desc. Returns (tp, fp, fn, matched_gt_mask)."""
    if len(pred) == 0:
        return 0, 0, len(gt), np.zeros(len(gt), bool)
    order = np.argsort(-pred_conf)
    ious = _iou_matrix(pred[order], gt)
    gt_used = np.zeros(len(gt), bool)
    tp = fp = 0
    for i in range(len(order)):
        if len(gt) == 0:
            fp += 1
            continue
        j = int(np.argmax(ious[i]))
        if ious[i, j] >= iou_thr and not gt_used[j]:
            gt_used[j] = True
            tp += 1
        else:
            fp += 1
    return tp, fp, int((~gt_used).sum()), gt_used


def _draw(img, gt, pred, pred_conf, gt_used, pred_tp_mask, view_long_side: int = 1600):
    """Draw GT/pred boxes, then downscale to a legible size.

    Source frames are 4K; a 2px line vanishes at any review zoom. So line
    thickness and font scale with the frame, and the saved QA image is
    downscaled (longest side ~1600) AFTER drawing thick lines so they survive.
    Colours (BGR): red=missed GT, orange=matched GT, green=TP pred, yellow=FP.
    """
    import cv2

    out = img.copy()
    h, w = out.shape[:2]
    thick = max(2, round(max(h, w) / 450))   # ~9px on a 3840-wide frame
    fs = max(0.5, max(h, w) / 2600)          # ~1.5 on 4K

    for k, (x1, y1, x2, y2) in enumerate(gt.astype(int)):
        color = (0, 140, 255) if gt_used[k] else (0, 0, 255)  # orange / red
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
    for k, (x1, y1, x2, y2) in enumerate(pred.astype(int)):
        color = (0, 200, 0) if pred_tp_mask[k] else (0, 215, 255)  # green / yellow
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        cv2.putText(out, f"{pred_conf[k]:.2f}", (x1, max(int(fs * 22), y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, color, thick, cv2.LINE_AA)

    # Legend strip (top-left).
    legend = [("GT miss", (0, 0, 255)), ("GT hit", (0, 140, 255)),
              ("pred TP", (0, 200, 0)), ("pred FP", (0, 215, 255))]
    y = int(fs * 36)
    for txt, col in legend:
        cv2.putText(out, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, fs, col,
                    thick, cv2.LINE_AA)
        y += int(fs * 40)

    scale = view_long_side / max(h, w)
    if scale < 1.0:
        out = cv2.resize(out, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--dataset", required=True, help="dataset.yaml (absolute)")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.10, help="operating-point conf")
    ap.add_argument("--nms-iou", type=float, default=0.60)
    ap.add_argument("--iou-match", type=float, default=0.50)
    ap.add_argument("--device", default="0")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--ledger", default="runs/ledger.csv")
    ap.add_argument("--qa-dir", default=None, help="default runs/<run-id>/qa")
    ap.add_argument("--max-qa", type=int, default=80)
    ap.add_argument("--skip-map", action="store_true", help="skip Ultralytics val mAP")
    # ledger metadata
    ap.add_argument("--model", default="")
    ap.add_argument("--aug", default="")
    ap.add_argument("--fold", default="")
    ap.add_argument("--epochs", default="")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    from ultralytics import YOLO

    ds_path = Path(args.dataset).resolve()
    ds = _read_yaml_min(ds_path)
    root = Path(ds.get("path", ds_path.parent))
    val_list = root / ds.get("val", "val_images.txt")
    images = [Path(p) for p in val_list.read_text(encoding="utf-8").splitlines() if p.strip()]
    if not images:
        LOGGER.error("Empty val list: %s", val_list)
        return 1
    LOGGER.info("Val images: %d (from %s)", len(images), val_list)

    qa_dir = Path(args.qa_dir) if args.qa_dir else Path("runs") / args.run_id / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    tot_tp = tot_fp = tot_fn = tot_gt = tot_pred = 0
    per_video: dict[str, list[int]] = {}
    frame_rows = []  # (severity, name, img, gt, pred, conf, gt_used, tp_mask)

    for img_path in images:
        img = imread(img_path)
        if img is None:
            LOGGER.warning("Unreadable image, skipped: %s", img_path)
            continue
        h, w = img.shape[:2]
        gt = _read_gt_boxes(_label_path_for(img_path), w, h)

        res = model.predict(source=str(img_path), conf=args.conf, iou=args.nms_iou,
                             imgsz=args.imgsz, device=args.device, verbose=False)[0]
        if res.boxes is not None and len(res.boxes):
            pred = res.boxes.xyxy.cpu().numpy().astype(np.float32)
            pconf = res.boxes.conf.cpu().numpy().astype(np.float32)
        else:
            pred = np.zeros((0, 4), np.float32)
            pconf = np.zeros((0,), np.float32)

        tp, fp, fn, gt_used = _match(pred, pconf, gt, args.iou_match)
        # which preds are TP (for colouring): re-derive by greedy on conf desc
        tp_mask = np.zeros(len(pred), bool)
        if len(pred) and len(gt):
            order = np.argsort(-pconf)
            ious = _iou_matrix(pred[order], gt)
            used = np.zeros(len(gt), bool)
            for i in range(len(order)):
                j = int(np.argmax(ious[i])) if len(gt) else -1
                if j >= 0 and ious[i, j] >= args.iou_match and not used[j]:
                    used[j] = True
                    tp_mask[order[i]] = True

        tot_tp += tp; tot_fp += fp; tot_fn += fn
        tot_gt += len(gt); tot_pred += len(pred)
        vid = img_path.parent.name
        pv = per_video.setdefault(vid, [0, 0, 0, 0])
        pv[0] += tp; pv[1] += fp; pv[2] += fn; pv[3] += len(gt)

        severity = fn * 2 + fp  # worst frames first
        frame_rows.append((severity, img_path.name, vid, img, gt, pred, pconf, gt_used, tp_mask))

    # Render QA: worst frames first, capped.
    frame_rows.sort(key=lambda r: -r[0])
    for sev, name, vid, img, gt, pred, pconf, gt_used, tp_mask in frame_rows[: args.max_qa]:
        out = _draw(img, gt, pred, pconf, gt_used, tp_mask)
        imwrite(qa_dir / f"sev{sev:03d}_{vid}_{name}", out)

    recall = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) else 0.0
    precision = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) else 0.0
    n_img = len(frame_rows)
    avg_det = tot_pred / n_img if n_img else 0.0

    map50 = map5095 = ""
    if not args.skip_map:
        try:
            # workers=0: Ultralytics' val dataloader, like train, can
            # intermittently deadlock with workers>0 on Windows. The val set
            # is tiny so single-process loading costs nothing.
            vr = model.val(data=str(ds_path), imgsz=args.imgsz, device=args.device,
                           conf=0.001, iou=args.nms_iou, workers=0, verbose=False)
            map50 = round(float(vr.box.map50), 4)
            map5095 = round(float(vr.box.map), 4)
        except Exception as e:  # noqa: BLE001 - mAP is a nice-to-have, never fatal
            LOGGER.warning("Ultralytics val (mAP) failed, continuing: %s", e)

    summary = {
        "run_id": args.run_id, "conf": args.conf, "iou_match": args.iou_match,
        "n_images": n_img, "n_gt": tot_gt, "tp": tot_tp, "fp": tot_fp, "fn": tot_fn,
        "recall": round(recall, 4), "precision": round(precision, 4),
        "map50": map50, "map5095": map5095, "avg_det_per_frame": round(avg_det, 2),
        "per_video": {k: {"tp": v[0], "fp": v[1], "fn": v[2], "n_gt": v[3],
                          "recall": round(v[0] / v[3], 4) if v[3] else 0.0}
                      for k, v in sorted(per_video.items())},
        "weights": str(Path(args.weights).resolve()), "qa_dir": str(qa_dir.resolve()),
    }
    summ_path = Path("runs") / args.run_id / "eval_summary.json"
    summ_path.parent.mkdir(parents=True, exist_ok=True)
    summ_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    ledger = Path(args.ledger)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    new = not ledger.exists()
    with ledger.open("a", encoding="utf-8", newline="") as f:
        wri = csv.writer(f)
        if new:
            wri.writerow(LEDGER_COLUMNS)
        wri.writerow([
            args.run_id, args.model, args.imgsz, args.aug, args.fold, args.epochs,
            args.conf, args.iou_match, n_img, tot_gt, tot_tp, tot_fp, tot_fn,
            round(recall, 4), round(precision, 4), map50, map5095,
            round(avg_det, 2), str(Path(args.weights).resolve()), args.notes,
        ])

    print("\n==== eval_qa summary ====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nQA images: {qa_dir.resolve()}  (worst frames first: sevNNN_*)")
    print(f"Ledger:    {ledger.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
