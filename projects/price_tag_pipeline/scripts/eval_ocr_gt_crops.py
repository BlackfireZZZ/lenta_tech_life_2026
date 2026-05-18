#!/usr/bin/env python3
"""Isolated OCR-on-GT-crops benchmark.

Measures the recognition chain (OCR by default) **without** the detector or
tracker in the loop: each crop is cut straight from a ground-truth bbox, so
the only thing under test is crop-rectify → OCR/VLM → parser → 29-column row.

Why this exists
---------------
``docs/runbooks/inference-current-issues.md`` and
``docs/ml-quality-improvement-plan.md`` both name "build a GT-crop OCR
benchmark from the organizer CSVs" as the next step: without isolated,
repeatable measurement on real crops every prompt/backend tweak is blind
(see also the project memory ``verify-dont-assert``).

Metric fidelity
---------------
There is exactly one hackathon-faithful scorer in the repo —
``scripts/eval_hack_csv.py`` (23 graded columns, "нет" compared as a literal
string so the task §5.3 нет-vs-empty rule is enforced, ≥80%-of-fields-per-tag
then share over all GT tags). This script does **not** reimplement it: it
imports ``_field_match`` / ``_per_tag_accuracy`` / ``DEFAULT_GRADED_FIELDS``
and applies them 1:1 to ``(predicted_row_from_crop, gt_row)`` pairs. Matching
is by construction (one crop per GT row) — IoU/timestamp matching is the
detector/tracker branch's concern, deliberately out of the OCR loop here.

The predicted row is built through the *production* renderer
(``submission.hack_row_from_tag_dict``), so the benchmark scores exactly the
bytes a real submission would emit for that crop.

Usage
-----
    .venv/Scripts/python.exe \\
      projects/price_tag_pipeline/scripts/eval_ocr_gt_crops.py \\
      --backend tesseract --limit 20            # fast plumbing sanity

    .venv/Scripts/python.exe \\
      projects/price_tag_pipeline/scripts/eval_ocr_gt_crops.py \\
      --config projects/price_tag_pipeline/configs/hq.yaml \\
      --backend paddle_vl                       # real local-GPU VLM run

Frames + GT come from the **main checkout's** ``data/processed`` (a worktree's
own ``data/processed`` is gitignored/empty — see docs/runbooks/venv-setup.md).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional

THIS = Path(__file__).resolve().parent
SRC = THIS.parent / "src"
for _p in (str(SRC), str(THIS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

from price_tag_pipeline import cv_io  # noqa: E402
from price_tag_pipeline.config import RecognitionConfig, load_config  # noqa: E402
from price_tag_pipeline.recognition.chain import build_recognition_chain  # noqa: E402
from price_tag_pipeline.rectifier import build_rectifier  # noqa: E402
from price_tag_pipeline.submission import hack_row_from_tag_dict  # noqa: E402
from price_tag_pipeline.types import HACK_EXTRA_FIELDS, Detection  # noqa: E402

# The single source of truth for the hackathon metric. Importing the module is
# side-effect free (its main() is __main__-guarded).
import eval_hack_csv as ehc  # noqa: E402

LOGGER = logging.getLogger("eval_ocr_gt_crops")

REPO_ROOT = Path("E:/Hackatons/lenta_tech_life_2026")
DEFAULT_FRAMES_ROOT = REPO_ROOT / "data" / "processed" / "frames"
DEFAULT_GT_ROOT = REPO_ROOT / "data" / "processed" / "gt_e2e"
ALL_VIDEOS = ("25_12-20", "25_2-10", "26_12-20", "43_15", "49_5")

# Columns that are GTIN/numeric-identity: the repo scorer compares them as
# exact normalized strings, so GT grouping spaces ("4 607124 143901") sink an
# otherwise-perfect read. We additionally report a digit-normalized view so
# the *true* OCR signal is visible next to the strict (submission) score.
_DIGIT_IDENTITY_FIELDS = ("barcode", "qr_code_barcode")


def _digits(s: Any) -> str:
    return "".join(ch for ch in str(s or "") if ch.isdigit())


def _gt_bbox_to_xyxy(row: dict[str, Any], w: int, h: int) -> Optional[tuple[int, int, int, int]]:
    bb = row.get("bbox")
    if not bb or len(bb) != 4:
        return None
    x1, y1, x2, y2 = (float(v) for v in bb)
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1i, y1i = max(0, int(round(x1))), max(0, int(round(y1)))
    x2i, y2i = min(w - 1, int(round(x2))), min(h - 1, int(round(y2)))
    if x2i - x1i < 4 or y2i - y1i < 4:
        return None
    return (x1i, y1i, x2i, y2i)


def _parsed_to_pred_row(parsed, filename: str) -> dict[str, str]:
    """ParsedTag → 29-column row via the *production* renderer.

    Mirrors ``FinalTag.to_dict()`` shape so ``hack_row_from_tag_dict`` maps
    ``regular_price→price_default`` / ``loyalty_price→price_card`` and passes
    every HACK_EXTRA_FIELDS key through exactly as a real submission would.
    """
    d: dict[str, Any] = {
        "bbox": [0, 0, 0, 0],
        "timestamp_s": 0.0,
        "regular_price": parsed.regular_price,
        "loyalty_price": parsed.loyalty_price,
        "product_name": parsed.product_name,
        "weight_value": parsed.weight_value,
        "weight_unit": parsed.weight_unit.value if parsed.weight_unit else None,
        "price_per_unit_value": parsed.price_per_unit_value,
        "price_per_unit_unit": parsed.price_per_unit_unit,
        "promo_flag": bool(parsed.promo_flag),
        "currency": parsed.currency,
    }
    for k in HACK_EXTRA_FIELDS:
        d[k] = parsed.extra_fields.get(k)
    return hack_row_from_tag_dict(d, filename)


def _best_result(results):
    """Single-crop OCR signal: the highest-confidence reading.

    Cross-frame weighted voting (the aggregator) is deliberately excluded —
    this benchmark isolates what one crop yields. ``found`` reads are
    preferred over empty ones at equal confidence.
    """
    if not results:
        return None
    return max(results, key=lambda r: (r.found, r.confidence))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(THIS.parent / "configs" / "balanced.yaml"))
    ap.add_argument("--backend", default=None, help="override ocr.backend (noop/tesseract/paddle/paddle_vl/glm_ocr/...)")
    ap.add_argument("--vlm-model", default=None, help="override ocr.vlm_model")
    ap.add_argument("--frames-root", default=str(DEFAULT_FRAMES_ROOT))
    ap.add_argument("--gt-root", default=str(DEFAULT_GT_ROOT))
    ap.add_argument("--videos", default=",".join(ALL_VIDEOS), help="comma-separated video ids")
    ap.add_argument("--limit", type=int, default=0, help="cap rows per video (0 = all)")
    ap.add_argument("--crop-mode", choices=("rectifier", "tight"), default="rectifier",
                    help="rectifier = pipeline-consistent padded crop; tight = raw GT bbox")
    ap.add_argument("--chain", choices=("ocr", "full"), default="ocr",
                    help="ocr = OCR only (isolate OCR); full = QR+barcode+OCR")
    ap.add_argument("--numeric-tol", type=float, default=0.01)
    ap.add_argument("--tag-pass-threshold", type=float, default=0.80)
    ap.add_argument("--dump-csv", default=None, help="also write the predicted hackathon CSV here")
    ap.add_argument("--dump-crops", default=None, help="dir to save crops for eyeballing")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    if args.backend:
        cfg = dataclasses.replace(cfg, ocr=dataclasses.replace(cfg.ocr, backend=args.backend.lower().strip()))
    if args.vlm_model:
        cfg = dataclasses.replace(cfg, ocr=dataclasses.replace(cfg.ocr, vlm_model=args.vlm_model))
    if args.chain == "ocr":
        cfg = dataclasses.replace(
            cfg, recognition=RecognitionConfig(enable_qr=False, enable_barcode=False, enable_ocr=True)
        )

    LOGGER.info("OCR backend=%s | chain=%s | crop=%s", cfg.ocr.backend, args.chain, args.crop_mode)
    rectifier = build_rectifier(cfg.rectifier)
    chain = build_recognition_chain(cfg)

    frames_root = Path(args.frames_root)
    gt_root = Path(args.gt_root)
    videos = [v.strip() for v in args.videos.split(",") if v.strip()]
    dump_crops = Path(args.dump_crops) if args.dump_crops else None

    pairs: list[tuple[dict[str, str], dict[str, Any]]] = []
    pred_rows_for_csv: list[dict[str, str]] = []
    n_rows = n_missing_frame = n_degenerate = n_ocr_empty = 0
    digit_hits = Counter()       # field -> exact digit match count
    digit_total = Counter()      # field -> GT rows with a real digit value
    t0 = time.time()

    for vid in videos:
        gt_path = gt_root / f"{vid}.jsonl"
        if not gt_path.exists():
            LOGGER.warning("GT not found, skipping: %s", gt_path)
            continue
        rows = [json.loads(ln) for ln in gt_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if args.limit:
            rows = rows[: args.limit]

        frame_cache: dict[int, Optional[np.ndarray]] = {}
        for row in rows:
            n_rows += 1
            fidx = int(row["frame_idx"])
            if fidx not in frame_cache:
                fp = frames_root / vid / f"{fidx:06d}.jpg"
                frame_cache[fidx] = cv_io.imread(str(fp))
            frame = frame_cache[fidx]
            if frame is None:
                n_missing_frame += 1
                continue
            h, w = frame.shape[:2]
            box = _gt_bbox_to_xyxy(row, w, h)
            if box is None:
                n_degenerate += 1
                continue

            if args.crop_mode == "tight":
                crop_img = frame[box[1]:box[3], box[0]:box[2]].copy()
            else:
                det = Detection(
                    frame_idx=fidx, timestamp_s=0.0, bbox_xyxy=box,
                    confidence=1.0, class_id=0, class_name="price_tag", track_id=0,
                )
                cand = rectifier.rectify(frame, det)
                if cand is None or cand.image.size == 0:
                    n_degenerate += 1
                    continue
                crop_img = cand.image

            if dump_crops is not None:
                cv_io.imwrite(str(dump_crops / vid / f"{fidx:06d}_{box[0]}_{box[1]}.jpg"), crop_img)

            results = chain.decode(crop_img)
            best = _best_result(results)
            if best is None or not best.found:
                n_ocr_empty += 1
            parsed = best.parsed if best is not None else None

            if parsed is None:
                # Empty prediction still scored: it's an honest OCR miss and
                # most "нет" GT fields will (correctly) be counted as lost.
                from price_tag_pipeline.types import ParsedTag
                parsed = ParsedTag()

            pred_row = _parsed_to_pred_row(parsed, row.get("filename", f"{vid}.mp4"))
            pred_row["filename"] = row.get("filename", f"{vid}.mp4")
            for k in ("frame_timestamp", "x_min", "y_min", "x_max", "y_max"):
                pred_row[k] = str(row.get(k, ""))
            pairs.append((pred_row, row))
            pred_rows_for_csv.append(pred_row)

            for f in _DIGIT_IDENTITY_FIELDS:
                g = _digits(row.get(f, ""))
                if g and row.get(f, "").strip().lower() != "нет":
                    digit_total[f] += 1
                    if _digits(pred_row.get(f, "")) == g:
                        digit_hits[f] += 1

    # ---- score: reuse the repo's faithful hackathon scorer, 1:1 ----------
    fields = ehc.DEFAULT_GRADED_FIELDS
    field_correct = Counter()
    field_considered = Counter()
    net_leak = Counter()       # GT == "нет" but pred empty (task §5.3 loss)
    net_total = Counter()      # GT == "нет" rows considered per field
    per_tag_scores: list[float] = []

    for pred, gt in pairs:
        considered = correct = 0
        for f in fields:
            gt_v = gt.get(f, "")
            gt_n = ehc._norm_text(gt_v)
            if not gt_n:
                continue
            considered += 1
            field_considered[f] += 1
            ok = ehc._field_match(f, pred.get(f, ""), gt_v, numeric_tol=args.numeric_tol)
            if ok:
                correct += 1
                field_correct[f] += 1
            if gt_n == "нет":
                net_total[f] += 1
                if not ehc._norm_text(pred.get(f, "")):
                    net_leak[f] += 1
        per_tag_scores.append(correct / considered if considered else 0.0)

    gt_total = sum(
        1 for v in videos
        for ln in (gt_root / f"{v}.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ) if not args.limit else len(pairs)
    passed = sum(1 for s in per_tag_scores if s >= args.tag_pass_threshold)
    mean_acc = sum(per_tag_scores) / len(per_tag_scores) if per_tag_scores else 0.0
    dt = time.time() - t0

    print("\n" + "=" * 64)
    print(f"OCR-on-GT-crops benchmark  |  backend={cfg.ocr.backend}  chain={args.chain}  crop={args.crop_mode}")
    print("=" * 64)
    print(f"GT rows seen:            {n_rows}")
    print(f"  missing frame:         {n_missing_frame}")
    print(f"  degenerate crop:       {n_degenerate}")
    print(f"  scored crops:          {len(pairs)}")
    print(f"  OCR returned nothing:  {n_ocr_empty}  ({n_ocr_empty / max(1, len(pairs)):.1%} of scored)")
    print(f"time: {dt:.1f}s  ({dt / max(1, len(pairs)):.2f}s/crop)")
    print("-" * 64)
    print(f"Mean scored-tag accuracy:                {mean_acc:.4f}")
    print(f"Share of GT tags with acc >= {args.tag_pass_threshold:.2f}:  "
          f"{passed}/{gt_total} = {passed / max(1, gt_total):.4f}   <<< headline")
    print("-" * 64)
    print("Per-field accuracy (worst first)  [correct/considered]:")
    rep = sorted(
        ((f, field_correct[f] / field_considered[f] if field_considered[f] else 0.0,
          field_correct[f], field_considered[f]) for f in fields),
        key=lambda x: x[1],
    )
    for f, acc, c, n in rep:
        leak = net_leak.get(f, 0)
        leak_s = f"  (нет-leak {leak}/{net_total.get(f, 0)})" if leak else ""
        print(f"  {f:26s} {acc:6.3f}  [{c:4d}/{n:4d}]{leak_s}")
    print("-" * 64)
    tot_leak = sum(net_leak.values())
    print(f'"нет"-vs-empty loss: {tot_leak} field-instances lost purely because the '
          f"parser never emits \"нет\" (task §5.3).")
    print("Digit-normalized identity (true OCR signal vs the strict string scorer):")
    for f in _DIGIT_IDENTITY_FIELDS:
        if digit_total[f]:
            print(f"  {f:26s} exact-digits {digit_hits[f]}/{digit_total[f]} = "
                  f"{digit_hits[f] / digit_total[f]:.3f}")
    print("=" * 64)

    if args.dump_csv:
        from price_tag_pipeline.submission import hack_rows_to_csv_text
        out = Path(args.dump_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(hack_rows_to_csv_text(pred_rows_for_csv), encoding="utf-8")
        LOGGER.info("Wrote predicted CSV (%d rows): %s", len(pred_rows_for_csv), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
