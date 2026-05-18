"""Measure QR(DataMatrix) + barcode(GS1 DataBar) decode rate **per price-tag
crop**, the way the production pipeline sees it.

Why crops, not full photos: a 50 MP shelf photo decodes in ~7 min and also
picks up unrelated product-package barcodes. The detector hands the pipeline a
small tag crop; that is the realistic, fast, unambiguous unit. Tag boxes come
from `friends_labels/all_candidates.csv` (YOLO `price_tag` detections on the
full-res photos).

EXIF caveat: the CSV boxes are in the EXIF-upright coordinate space
(image_width<image_height = portrait), but OpenCV decodes raw pixels
(landscape). We load through PIL `exif_transpose` so crops line up with the
boxes; verified by comparing (W,H) against the CSV.

Usage:
    .venv/Scripts/python.exe projects/price_tag_pipeline/scripts/eval_qr.py \
        friends_labels/good_for_qr \
        --candidates friends_labels/all_candidates.csv \
        [--limit N] [--max-per-image M] [--no-wechat] [--fast]
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from price_tag_pipeline.recognition.qr_engine import (  # noqa: E402
    CascadeConfig,
    QRBarcodeEngine,
)

_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _load_exif_upright_bgr(path: Path) -> "np.ndarray | None":
    """Load an image with EXIF orientation applied, as a BGR ndarray.

    Matches the coordinate space of the YOLO boxes in all_candidates.csv.
    """
    try:
        from PIL import Image, ImageOps
    except Exception:
        print("!! Pillow required for EXIF-correct cropping")
        return None
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            rgb = np.asarray(im)
    except Exception as exc:
        print(f"!! cannot open {path.name}: {exc}")
        return None
    return rgb[:, :, ::-1].copy()  # RGB -> BGR


def _read_boxes(candidates_csv: Path) -> dict[str, list[dict]]:
    by_image: dict[str, list[dict]] = defaultdict(list)
    with open(candidates_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("class_name") or "").strip() != "price_tag":
                continue
            try:
                box = {
                    "x_min": int(float(row["x_min"])),
                    "y_min": int(float(row["y_min"])),
                    "x_max": int(float(row["x_max"])),
                    "y_max": int(float(row["y_max"])),
                    "w": int(float(row["image_width"])),
                    "h": int(float(row["image_height"])),
                    "conf": float(row.get("confidence") or 0.0),
                }
            except (KeyError, ValueError):
                continue
            by_image[row["image_key"]].append(box)
    return by_image


def _crop(img: np.ndarray, b: dict, pad: float) -> np.ndarray:
    H, W = img.shape[:2]
    px = int((b["x_max"] - b["x_min"]) * pad)
    py = int((b["y_max"] - b["y_min"]) * pad)
    x0, y0 = max(0, b["x_min"] - px), max(0, b["y_min"] - py)
    x1, y1 = min(W, b["x_max"] + px), min(H, b["y_max"] + py)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return np.empty((0, 0, 3), np.uint8)
    return img[y0:y1, x0:x1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--candidates", type=Path,
                    default=Path("friends_labels/all_candidates.csv"))
    ap.add_argument("--limit", type=int, default=0, help="max images")
    ap.add_argument("--max-per-image", type=int, default=0,
                    help="cap tag boxes per image (0 = all)")
    ap.add_argument("--pad", type=float, default=0.06,
                    help="fraction of box size to pad the crop")
    ap.add_argument("--wechat", action="store_true",
                    help="also run WeChat QR (useless for Lenta DataMatrix; "
                         "only for QR-bearing inputs)")
    ap.add_argument("--fast", action="store_true",
                    help="stop at first symbol per crop (production mode) "
                         "instead of the exhaustive ceiling")
    args = ap.parse_args()

    boxes_by_image = _read_boxes(args.candidates)
    files = sorted(p for p in args.folder.iterdir()
                   if p.suffix.lower() in _EXTS and p.name in boxes_by_image)
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"no images in {args.folder} matching {args.candidates}")
        return 1

    decoders = ("zxing", "pyzbar")
    if args.wechat:
        decoders = decoders + ("wechat",)
    engine = QRBarcodeEngine(CascadeConfig(
        decoders=decoders,
        exhaustive=not args.fast,
        stop_when="any" if args.fast else "both",
        max_side=2200,
    ))
    print(f"backends     : {engine.b.available()}")
    print(f"mode         : {'FAST (stop@first)' if args.fast else 'EXHAUSTIVE'}")
    print(f"images       : {len(files)} from {args.folder}\n")

    n_tags = n_1d = n_1d_valid = n_2d = n_any = 0
    stage_hits: Counter = Counter()
    decoder_hits: Counter = Counter()
    fmt_hits: Counter = Counter()
    samples: list[str] = []
    t_total = 0.0

    for p in files:
        img = _load_exif_upright_bgr(p)
        if img is None:
            continue
        H, W = img.shape[:2]
        boxes = boxes_by_image[p.name]
        csv_w, csv_h = boxes[0]["w"], boxes[0]["h"]
        orient = "OK" if (W, H) == (csv_w, csv_h) else f"MISMATCH({csv_w}x{csv_h})"
        if args.max_per_image:
            boxes = sorted(boxes, key=lambda b: -b["conf"])[: args.max_per_image]

        img_1d = img_2d = img_any = 0
        t0 = time.perf_counter()
        for b in boxes:
            crop = _crop(img, b, args.pad)
            if crop.size == 0:
                continue
            syms = engine.decode(crop)
            n_tags += 1
            kinds = {s.kind for s in syms}
            has_valid_1d = any(s.kind == "1d" and s.checksum_ok is not False
                               for s in syms)
            img_1d += "1d" in kinds
            img_2d += "2d" in kinds
            img_any += bool(kinds)
            n_1d += "1d" in kinds
            n_1d_valid += has_valid_1d
            n_2d += "2d" in kinds
            n_any += bool(kinds)
            for s in syms:
                stage_hits[s.stage] += 1
                decoder_hits[s.decoder] += 1
                fmt_hits[s.fmt] += 1
                if len(samples) < 24:
                    chk = "" if s.checksum_ok is None else (
                        " chk=OK" if s.checksum_ok else " chk=BAD")
                    g = f" gtin={s.gtin}" if s.gtin else ""
                    samples.append(f"  {p.name[:18]} {s.kind} {s.fmt:<14}"
                                   f" {s.decoder:<10} @{s.stage:<18}{chk}{g}"
                                   f"  {s.text!r}")
        dt = time.perf_counter() - t0
        t_total += dt
        print(f"[{orient:>14}] {p.name}: {len(boxes)} tags  "
              f"1d={img_1d} 2d={img_2d} any={img_any}  ({dt:.1f}s)")

    if n_tags == 0:
        print("no decodable tag crops")
        return 1
    print("\n----- sample decodes -----")
    print("\n".join(samples))
    print("\n==================== SUMMARY ====================")
    print(f"tag crops               : {n_tags}")
    print(f"≥1 ANY symbol           : {n_any}/{n_tags}  ({n_any / n_tags:.0%})")
    print(f"≥1 1D barcode (DataBar) : {n_1d}/{n_tags}  ({n_1d / n_tags:.0%})")
    print(f"  ...checksum-valid     : {n_1d_valid}/{n_tags}  ({n_1d_valid / n_tags:.0%})")
    print(f"≥1 2D (DataMatrix/QR)   : {n_2d}/{n_tags}  ({n_2d / n_tags:.0%})")
    print(f"avg time/tag            : {t_total / n_tags:.2f}s")
    print(f"\nformats     : {fmt_hits.most_common()}")
    print(f"by decoder  : {decoder_hits.most_common()}")
    print(f"by stage    : {stage_hits.most_common(12)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
