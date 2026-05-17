#!/usr/bin/env python3
"""External retail dataset intake and task-dataset builder.

The repo keeps external data under ``dataset_research`` until it is audited.
This script makes that intake reproducible:

* records which requested datasets are available locally and which need keys;
* converts compatible raw formats into task-specific YOLO-style research sets;
* keeps product/object datasets separate from price-tag detector data;
* creates contact sheets with boxes for visual QA.

The important rule is that no external dataset is treated as final validation.
These outputs are candidate training/pretraining sets only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml
from PIL import Image, ImageDraw


DEFAULT_DATASET_RESEARCH_ROOT = Path(__file__).resolve().parents[4] / "dataset_research"
ROOT = Path(os.environ.get("DATASET_RESEARCH_ROOT", DEFAULT_DATASET_RESEARCH_ROOT)).resolve()
TASK_ROOT = ROOT / "task_datasets"


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    source_url: str
    raw_dir: Path
    expected_hint: str
    role: str
    access_note: str


DATASETS: dict[str, DatasetSpec] = {
    "openfoodfacts": DatasetSpec(
        dataset_id="openfoodfacts",
        source_url="https://huggingface.co/datasets/openfoodfacts/price-tag-detection",
        raw_dir=ROOT / "openfoodfacts_price_tag_detection" / "raw",
        expected_hint="train-00000-of-00001.parquet + val-00000-of-00001.parquet",
        role="price_tag detector candidates + OCR full-tag crops",
        access_note="Already downloadable from Hugging Face; current raw parquet is present in this workspace.",
    ),
    "sovar_roboflow": DatasetSpec(
        dataset_id="sovar_roboflow",
        source_url="https://universe.roboflow.com/sovar/price-tag-detection-r5jlv/dataset/2",
        raw_dir=ROOT / "sovar_price_tag_detection" / "raw",
        expected_hint="Roboflow YOLOv8 export with data.yaml and train/valid/test folders",
        role="small on-domain price_tag detector candidate; must de-augment / QA",
        access_note="Needs ROBOFLOW_API_KEY or a manual Roboflow YOLOv8 export.",
    ),
    "hitl_supermarket_shelves": DatasetSpec(
        dataset_id="hitl_supermarket_shelves",
        source_url="https://www.kaggle.com/datasets/humansintheloop/supermarket-shelves-dataset",
        raw_dir=ROOT / "hitl_supermarket_shelves" / "raw",
        expected_hint="Kaggle folder with shelf images + Supervisely JSON annotations",
        role="Price boxes -> price_tag_broad; Product boxes -> product context/hard negatives",
        access_note="Needs Kaggle credentials or a manual Kaggle download.",
    ),
    "sku110k": DatasetSpec(
        dataset_id="sku110k",
        source_url="https://docs.ultralytics.com/datasets/detect/sku-110k/",
        raw_dir=ROOT / "sku110k" / "raw",
        expected_hint="SKU-110K/SKU110K_fixed with annotations_train.csv, annotations_val.csv, annotations_test.csv",
        role="object/product pretraining only; never map boxes to price_tag",
        access_note="Public mirror is large (~13.6 GB); academic/non-commercial source terms apply.",
    ),
}


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_yolo_boxes(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes: list[tuple[int, float, float, float, float]] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:])
        except ValueError:
            continue
        if 0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1:
            boxes.append((cls, cx, cy, w, h))
    return boxes


def _box_area(box: tuple[int, float, float, float, float]) -> float:
    return box[3] * box[4]


def _box_aspect(box: tuple[int, float, float, float, float]) -> float:
    return box[3] / box[4] if box[4] > 0 else math.inf


def _touches_edge(box: tuple[int, float, float, float, float], margin: float = 0.005) -> bool:
    _cls, cx, cy, w, h = box
    x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    return x1 <= margin or y1 <= margin or x2 >= 1 - margin or y2 >= 1 - margin


def _crop_quality(image_path: Path, box: tuple[int, float, float, float, float]) -> dict[str, float]:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    _cls, cx, cy, bw, bh = box
    x1 = max(0, int((cx - bw / 2) * width))
    y1 = max(0, int((cy - bh / 2) * height))
    x2 = min(width, int((cx + bw / 2) * width))
    y2 = min(height, int((cy + bh / 2) * height))
    if x2 <= x1 or y2 <= y1:
        return {"mean": 0.0, "std": 0.0, "dark_frac": 1.0, "bright_frac": 0.0}
    crop = image.crop((x1, y1, x2, y2)).resize((64, 64), Image.Resampling.BILINEAR)
    pixels = list(crop.getdata())
    gray = [0.299 * r + 0.587 * g + 0.114 * b for r, g, b in pixels]
    chroma = [max(r, g, b) - min(r, g, b) for r, g, b in pixels]
    mean = sum(gray) / len(gray)
    var = sum((v - mean) ** 2 for v in gray) / len(gray)
    dark_frac = sum(v < 35 for v in gray) / len(gray)
    bright_frac = sum(v > 225 for v in gray) / len(gray)
    chroma_mean = sum(chroma) / len(chroma)
    return {
        "mean": mean,
        "std": math.sqrt(var),
        "dark_frac": dark_frac,
        "bright_frac": bright_frac,
        "chroma_mean": chroma_mean,
    }


def _link_or_copy(src: Path, dst: Path) -> None:
    _safe_mkdir(dst.parent)
    if dst.exists():
        return
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def _write_yolo_dataset_yaml(out: Path, names: list[str], train: str = "train_images.txt", val: str = "val_images.txt") -> None:
    lines = [
        f"path: {out.resolve().as_posix()}",
        f"train: {train}",
        f"val: {val}",
        f"nc: {len(names)}",
        "names:",
    ]
    lines.extend(f"  {i}: {name}" for i, name in enumerate(names))
    (out / "dataset.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    _safe_mkdir(path.parent)
    fieldnames = sorted({k for row in rows for k in row.keys()}) if rows else ["image"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _draw_tile(image_path: Path, label_path: Path, title: str, tile: int) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    image.thumbnail((tile, tile), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (tile, tile + 28), (246, 246, 246))
    xoff = (tile - image.width) // 2
    yoff = 28
    canvas.paste(image, (xoff, yoff))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, tile - 1, tile + 27], outline=(210, 210, 210), width=1)
    draw.text((5, 7), title[:36], fill=(10, 10, 10))

    sx = image.width / orig_w
    sy = image.height / orig_h
    for _cls, cx, cy, bw, bh in _read_yolo_boxes(label_path):
        x1 = xoff + (cx - bw / 2) * orig_w * sx
        y1 = yoff + (cy - bh / 2) * orig_h * sy
        x2 = xoff + (cx + bw / 2) * orig_w * sx
        y2 = yoff + (cy + bh / 2) * orig_h * sy
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
    return canvas


def make_contact_sheet(dataset_dir: Path, out_path: Path, n: int = 80, cols: int = 5, tile: int = 220, seed: int = 20260517) -> None:
    manifest_path = dataset_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    rows = list(csv.DictReader(manifest_path.open(encoding="utf-8")))
    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = rows[: min(n, len(rows))]
    if not rows:
        return
    sheet_rows = (len(rows) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile, sheet_rows * (tile + 28)), (255, 255, 255))
    for i, row in enumerate(rows):
        image_path = dataset_dir / str(row["image"])
        label_path = dataset_dir / str(row["label"])
        title = f"{row.get('source_dataset', '')}:{row.get('source_image_id', Path(row['image']).stem)}"
        tile_image = _draw_tile(image_path, label_path, title, tile)
        sheet.paste(tile_image, ((i % cols) * tile, (i // cols) * (tile + 28)))
    _safe_mkdir(out_path.parent)
    sheet.save(out_path, quality=92)


def status() -> dict[str, object]:
    report: dict[str, object] = {}
    for dataset_id, spec in DATASETS.items():
        raw = spec.raw_dir
        if dataset_id == "openfoodfacts":
            present = (raw / "train-00000-of-00001.parquet").exists() and (raw / "val-00000-of-00001.parquet").exists()
        elif dataset_id == "sovar_roboflow":
            present = (raw / "data.yaml").exists() or any(raw.glob("**/data.yaml"))
        elif dataset_id == "hitl_supermarket_shelves":
            present = bool(list(raw.glob("**/*.json"))) and bool(list(raw.glob("**/*.jpg")) or list(raw.glob("**/*.png")))
        elif dataset_id == "sku110k":
            present = bool(list(raw.glob("**/annotations_train.csv")))
        else:
            present = raw.exists()
        report[dataset_id] = {
            "present": present,
            "raw_dir": raw.as_posix(),
            "source_url": spec.source_url,
            "expected_hint": spec.expected_hint,
            "role": spec.role,
            "access_note": spec.access_note,
        }

    report["credentials"] = {
        "ROBOFLOW_API_KEY": bool(os.environ.get("ROBOFLOW_API_KEY")),
        "kaggle_json": (Path.home() / ".kaggle" / "kaggle.json").exists(),
    }
    out = ROOT / "external_dataset_status.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def write_download_runbook() -> Path:
    out = ROOT / "external_dataset_download_runbook.md"
    text = """# External Dataset Download Runbook

Generated by `dataset_research/external_dataset_pipeline.py`.

All raw data should land in the dataset-specific `raw/` folder below
`dataset_research/`. Keep credentials outside the repo.

## OpenFoodFacts price-tag-detection

Already present in this workspace:

```bash
dataset_research/openfoodfacts_price_tag_detection/raw/
```

If re-downloading, use Hugging Face parquet files for `train` and `val`.

## SOVAR Roboflow price-tag-detection v2

Requires a Roboflow account/API key or manual export:

```bash
export ROBOFLOW_API_KEY=...
python -m pip install roboflow
python ultralytics/projects/price_tag_pipeline/scripts/fetch_external_datasets.py \\
  --download roboflow \\
  --rf-workspace sovar \\
  --rf-project price-tag-detection-r5jlv \\
  --rf-version 2
```

Then copy or move the YOLOv8 export so that `data.yaml` is under:

```bash
dataset_research/sovar_price_tag_detection/raw/
```

## HITL Supermarket Shelves

Requires Kaggle credentials (`~/.kaggle/kaggle.json`) or a browser download:

```bash
python -m pip install kaggle
kaggle datasets download -d humansintheloop/supermarket-shelves-dataset \\
  -p dataset_research/hitl_supermarket_shelves/raw --unzip
```

Expected: shelf images plus Supervisely JSON annotations with `Product` and
`Price` rectangle objects.

## SKU-110K

Large public mirror, about 13.6 GB after download/extract. Use only for
`object`/product pretraining, not price-tag positives.

```bash
python - <<'PY'
from ultralytics.utils.downloads import safe_download
safe_download(
    'https://github.com/ultralytics/assets/releases/download/v0.0.0/SKU-110K.zip',
    dir='dataset_research/sku110k/raw',
    unzip=True,
)
PY
```

Alternative official Ultralytics YAML downloads from the SKU-110K source mirror.
"""
    out.write_text(text, encoding="utf-8")
    return out


def build_openfoodfacts_tasks() -> list[Path]:
    src = ROOT / "openfoodfacts_price_tag_detection" / "clean_candidates_v0"
    manifest_path = src / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing OpenFoodFacts candidate manifest: {manifest_path}")

    strict_out = TASK_ROOT / "price_tag_detector" / "openfoodfacts_lenta_strict_candidates_v1"
    broad_out = TASK_ROOT / "price_tag_detector" / "openfoodfacts_price_tag_broad_v1"
    crop_out = TASK_ROOT / "ocr_tag_crops" / "openfoodfacts_full_tag_crops_v1"
    for out in (strict_out, broad_out, crop_out):
        _safe_mkdir(out)

    strict_rows: list[dict[str, object]] = []
    broad_rows: list[dict[str, object]] = []
    crop_rows: list[dict[str, object]] = []

    source_rows = list(csv.DictReader(manifest_path.open(encoding="utf-8")))
    for row in source_rows:
        image_src = src / str(row["image"])
        label_src = src / str(row["label"])
        boxes = _read_yolo_boxes(label_src)
        if not boxes:
            continue
        areas = sorted(_box_area(box) for box in boxes)
        aspects = [_box_aspect(box) for box in boxes if math.isfinite(_box_aspect(box))]
        max_area = max(areas)
        median_area = areas[len(areas) // 2]
        median_aspect = sorted(aspects)[len(aspects) // 2] if aspects else 0.0
        edge_count = sum(_touches_edge(box) for box in boxes)

        # Strict candidate: shelf-context-like, small/medium full tags, sane shape,
        # and no edge-cut labels. This is still a candidate tier, not ground truth.
        is_strict = (
            max_area <= 0.18
            and median_area <= 0.055
            and 1.25 <= median_aspect <= 6.0
            and edge_count == 0
            and (len(boxes) >= 2 or max_area <= 0.09)
        )
        target_out = strict_out if is_strict else broad_out
        image_rel = Path("images") / image_src.name
        label_rel = Path("labels") / label_src.name
        _link_or_copy(image_src, target_out / image_rel)
        _link_or_copy(label_src, target_out / label_rel)

        out_row = {
            "image": image_rel.as_posix(),
            "label": label_rel.as_posix(),
            "source_dataset": "openfoodfacts",
            "source_image_id": row.get("source_image_id", image_src.stem),
            "tier": "lenta_strict_candidate" if is_strict else "price_tag_broad",
            "split": row.get("split", ""),
            "n_labels": len(boxes),
            "max_area": f"{max_area:.6f}",
            "median_area": f"{median_area:.6f}",
            "median_aspect": f"{median_aspect:.6f}",
            "license": row.get("license", "CC-BY-SA-4.0"),
            "source": row.get("source", DATASETS["openfoodfacts"].source_url),
        }
        if is_strict:
            strict_rows.append(out_row)
        else:
            broad_rows.append(out_row)

        image = Image.open(image_src).convert("RGB")
        width, height = image.size
        for box_i, (_cls, cx, cy, bw, bh) in enumerate(boxes):
            if _box_area((_cls, cx, cy, bw, bh)) > 0.22:
                continue
            x1 = max(0, int((cx - bw / 2) * width))
            y1 = max(0, int((cy - bh / 2) * height))
            x2 = min(width, int((cx + bw / 2) * width))
            y2 = min(height, int((cy + bh / 2) * height))
            if x2 <= x1 or y2 <= y1:
                continue
            crop_name = f"{Path(image_src).stem}_box{box_i:02d}.jpg"
            crop_rel = Path("images") / crop_name
            crop = image.crop((x1, y1, x2, y2))
            _safe_mkdir((crop_out / crop_rel).parent)
            if not (crop_out / crop_rel).exists():
                crop.save(crop_out / crop_rel, quality=95)
            crop_rows.append({
                "image": crop_rel.as_posix(),
                "source_dataset": "openfoodfacts",
                "source_image_id": row.get("source_image_id", image_src.stem),
                "source_box_index": box_i,
                "tier": "ocr_full_tag_crop_candidate",
                "source_tier": "lenta_strict_candidate" if is_strict else "price_tag_broad",
                "license": row.get("license", "CC-BY-SA-4.0"),
                "source": row.get("source", DATASETS["openfoodfacts"].source_url),
            })

    for out, rows, tier in (
        (strict_out, strict_rows, "lenta_strict_candidate"),
        (broad_out, broad_rows, "price_tag_broad"),
    ):
        _write_manifest(out / "manifest.csv", rows)
        image_list = [str((out / row["image"]).resolve()) for row in rows]
        (out / "train_images.txt").write_text("\n".join(image_list) + ("\n" if image_list else ""), encoding="utf-8")
        (out / "val_images.txt").write_text("\n".join(image_list[: max(1, len(image_list) // 10)]) + ("\n" if image_list else ""), encoding="utf-8")
        _write_yolo_dataset_yaml(out, ["price_tag"])
        (out / "README.md").write_text(
            f"# OpenFoodFacts {tier} v1\n\n"
            "Task: price-tag detector research training candidate.\n\n"
            f"- Source rows: {len(source_rows)}\n"
            f"- Kept in this tier: {len(rows)}\n"
            "- Labels: YOLO class `0 price_tag`\n"
            "- Validation file is a smoke-check split only; use held-out real Lenta video for real validation.\n"
            "- Manual QA is required before using this as trusted training data.\n",
            encoding="utf-8",
        )
        if rows:
            make_contact_sheet(out, out / "qa_contact_sheet.jpg", n=100)

    _write_manifest(crop_out / "manifest.csv", crop_rows)
    (crop_out / "README.md").write_text(
        "# OpenFoodFacts full-tag OCR crops v1\n\n"
        "Task: OCR/field-extraction experiments on cropped price tags. These crops are not Lenta-layout ground truth; "
        "use them for generic readability, OCR smoke tests, and negative layout transfer checks.\n\n"
        f"- Crops: {len(crop_rows)}\n"
        "- No text labels are provided by the source dataset.\n",
        encoding="utf-8",
    )

    summary = {
        "source": "openfoodfacts clean_candidates_v0",
        "source_rows": len(source_rows),
        "lenta_strict_candidate_images": len(strict_rows),
        "price_tag_broad_images": len(broad_rows),
        "ocr_full_tag_crop_candidates": len(crop_rows),
        "strict_rule": "max_area<=0.18, median_area<=0.055, aspect 1.25..6.0, no edge-cut boxes, shelf-context proxy",
    }
    summary_path = ROOT / "openfoodfacts_price_tag_detection" / "processing_v1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return [strict_out, broad_out, crop_out]


def _find_roboflow_export(raw_dir: Path) -> Path | None:
    if (raw_dir / "data.yaml").exists():
        return raw_dir
    for path in raw_dir.glob("**/data.yaml"):
        return path.parent
    return None


def build_roboflow_yolo_tasks(dataset_id: str, raw_dir: Path, source_url: str) -> Path | None:
    export = _find_roboflow_export(raw_dir)
    if export is None:
        return None
    data = yaml.safe_load((export / "data.yaml").read_text(encoding="utf-8"))
    names_raw = data.get("names", {})
    if isinstance(names_raw, dict):
        names = {int(k): str(v) for k, v in names_raw.items()}
    else:
        names = {i: str(v) for i, v in enumerate(names_raw)}

    price_classes = {
        i for i, name in names.items()
        if any(token in name.lower().replace("-", "_") for token in ("price", "pricetag", "tag", "label"))
        and "digit" not in name.lower()
        and "product" not in name.lower()
    }
    out = TASK_ROOT / "price_tag_detector" / f"{dataset_id}_price_tag_broad_v1"
    rows: list[dict[str, object]] = []
    image_exts = {".jpg", ".jpeg", ".png", ".webp"}
    for split_name in ("train", "valid", "val", "test"):
        split_dir = export / split_name
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        if not images_dir.exists() or not labels_dir.exists():
            continue
        for image_path in sorted(p for p in images_dir.iterdir() if p.suffix.lower() in image_exts):
            label_path = labels_dir / f"{image_path.stem}.txt"
            selected: list[str] = []
            for cls, cx, cy, bw, bh in _read_yolo_boxes(label_path):
                box = (cls, cx, cy, bw, bh)
                if cls not in price_classes:
                    continue
                if not (0.00015 <= _box_area(box) <= 0.30):
                    continue
                if not (0.7 <= _box_aspect(box) <= 8.0):
                    continue
                selected.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            if not selected:
                continue
            image_rel = Path("images") / split_name / image_path.name
            label_rel = Path("labels") / split_name / f"{image_path.stem}.txt"
            _link_or_copy(image_path, out / image_rel)
            _safe_mkdir((out / label_rel).parent)
            (out / label_rel).write_text("\n".join(selected) + "\n", encoding="utf-8")
            rows.append({
                "image": image_rel.as_posix(),
                "label": label_rel.as_posix(),
                "split": split_name,
                "source_dataset": dataset_id,
                "source_image_id": image_path.stem,
                "tier": "price_tag_broad",
                "n_labels": len(selected),
                "license": "check Roboflow project page",
                "source": source_url,
            })

    if not rows:
        return out
    _write_manifest(out / "manifest.csv", rows)
    train_rows = [r for r in rows if r["split"] in {"train", "valid", "val"}]
    val_rows = [r for r in rows if r["split"] in {"valid", "val", "test"}] or rows[: max(1, len(rows) // 10)]
    (out / "train_images.txt").write_text("\n".join(str((out / r["image"]).resolve()) for r in train_rows) + "\n", encoding="utf-8")
    (out / "val_images.txt").write_text("\n".join(str((out / r["image"]).resolve()) for r in val_rows) + "\n", encoding="utf-8")
    _write_yolo_dataset_yaml(out, ["price_tag"])
    (out / "README.md").write_text(
        f"# {dataset_id} price_tag_broad v1\n\n"
        "Built from a Roboflow YOLO export. Only full price/label/tag-like classes are collapsed to `price_tag`; "
        "digit/product classes are deliberately excluded.\n\n"
        f"- Images: {len(rows)}\n"
        f"- Source classes: {names}\n"
        f"- Mapped source class ids: {sorted(price_classes)}\n",
        encoding="utf-8",
    )
    make_contact_sheet(out, out / "qa_contact_sheet.jpg", n=100)
    return out


def build_sovar_clean_task() -> Path | None:
    broad = TASK_ROOT / "price_tag_detector" / "sovar_roboflow_price_tag_broad_v1"
    manifest_path = broad / "manifest.csv"
    if not manifest_path.exists():
        return None
    out = TASK_ROOT / "price_tag_detector" / "sovar_roboflow_price_tag_clean_v1"
    rows: list[dict[str, object]] = []
    qa_rows: list[dict[str, object]] = []
    for row in csv.DictReader(manifest_path.open(encoding="utf-8")):
        image_path = broad / row["image"]
        label_path = broad / row["label"]
        kept_lines: list[str] = []
        rejected = 0
        for box in _read_yolo_boxes(label_path):
            quality = _crop_quality(image_path, box)
            area = _box_area(box)
            aspect = _box_aspect(box)
            # Roboflow SOVAR v2 contains many privacy/redaction rectangles. They
            # are visually unlike real price tags and are harmful detector positives.
            clean = (
                0.0002 <= area <= 0.22
                and 0.75 <= aspect <= 8.0
                and quality["dark_frac"] <= 0.35
                and quality["std"] >= 18.0
                and not (quality["mean"] < 45 and quality["dark_frac"] > 0.20)
                and not (quality["chroma_mean"] < 18 and quality["bright_frac"] < 0.08)
                and not (45 <= quality["mean"] <= 175 and quality["chroma_mean"] < 22 and quality["bright_frac"] < 0.12)
            )
            qa_rows.append({
                "image": row["image"],
                "label": row["label"],
                "source_image_id": row.get("source_image_id", ""),
                "area": f"{area:.6f}",
                "aspect": f"{aspect:.6f}",
                "mean": f"{quality['mean']:.3f}",
                "std": f"{quality['std']:.3f}",
                "dark_frac": f"{quality['dark_frac']:.3f}",
                "bright_frac": f"{quality['bright_frac']:.3f}",
                "chroma_mean": f"{quality['chroma_mean']:.3f}",
                "kept": int(clean),
            })
            if clean:
                _cls, cx, cy, bw, bh = box
                kept_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            else:
                rejected += 1
        if not kept_lines:
            continue
        image_rel = Path(row["image"])
        label_rel = Path(row["label"])
        _link_or_copy(image_path, out / image_rel)
        _safe_mkdir((out / label_rel).parent)
        (out / label_rel).write_text("\n".join(kept_lines) + "\n", encoding="utf-8")
        out_row = dict(row)
        out_row["n_labels"] = len(kept_lines)
        out_row["rejected_masked_or_low_quality_boxes"] = rejected
        out_row["tier"] = "price_tag_clean_candidate"
        rows.append(out_row)

    _write_manifest(out / "analysis" / "box_quality_report.csv", qa_rows)
    if not rows:
        return out
    _write_manifest(out / "manifest.csv", rows)
    train_rows = [r for r in rows if r["split"] in {"train", "valid", "val"}]
    val_rows = [r for r in rows if r["split"] in {"valid", "val", "test"}] or rows[: max(1, len(rows) // 10)]
    (out / "train_images.txt").write_text("\n".join(str((out / r["image"]).resolve()) for r in train_rows) + "\n", encoding="utf-8")
    (out / "val_images.txt").write_text("\n".join(str((out / r["image"]).resolve()) for r in val_rows) + "\n", encoding="utf-8")
    _write_yolo_dataset_yaml(out, ["price_tag"])
    (out / "README.md").write_text(
        "# SOVAR Roboflow price_tag_clean v1\n\n"
        "Filtered from `sovar_roboflow_price_tag_broad_v1` after visual QA showed many black/gray redaction boxes. "
        "This tier removes boxes with high dark-pixel fraction or very low crop texture, while preserving the broad tier for audit.\n\n"
        f"- Images: {len(rows)}\n"
        f"- Boxes: {sum(int(r['n_labels']) for r in rows)}\n"
        "- Use as candidate price-tag detector training data only; not final validation.\n",
        encoding="utf-8",
    )
    make_contact_sheet(out, out / "qa_contact_sheet.jpg", n=100)
    return out


def _voc_image_path(xml_path: Path) -> Path | None:
    root = ET.parse(xml_path).getroot()
    filename = root.findtext("filename")
    candidates: list[Path] = []
    if filename:
        candidates.extend(xml_path.parent.glob(filename))
        candidates.extend(xml_path.parent.parent.glob(f"**/{filename}"))
    candidates.extend(xml_path.with_suffix(ext) for ext in (".jpg", ".jpeg", ".png"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def build_hitl_voc_tasks(raw_dir: Path) -> tuple[Path | None, Path | None]:
    xml_paths = sorted(raw_dir.glob("**/*.xml"))
    if not xml_paths:
        return None, None
    price_out = TASK_ROOT / "price_tag_detector" / "hitl_supermarket_shelves_price_tag_broad_v1"
    product_out = TASK_ROOT / "product_context" / "hitl_supermarket_shelves_products_v1"
    price_rows: list[dict[str, object]] = []
    product_rows: list[dict[str, object]] = []

    for xml_path in xml_paths:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        image_path = _voc_image_path(xml_path)
        if image_path is None:
            continue
        width = int(root.findtext("size/width") or Image.open(image_path).size[0])
        height = int(root.findtext("size/height") or Image.open(image_path).size[1])
        price_lines: list[str] = []
        product_lines: list[str] = []
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip().lower()
            box = obj.find("bndbox")
            if box is None:
                continue
            x1 = float(box.findtext("xmin") or 0)
            y1 = float(box.findtext("ymin") or 0)
            x2 = float(box.findtext("xmax") or 0)
            y2 = float(box.findtext("ymax") or 0)
            x1, x2 = sorted((max(0.0, x1), min(float(width), x2)))
            y1, y2 = sorted((max(0.0, y1), min(float(height), y2)))
            if x2 <= x1 or y2 <= y1:
                continue
            cx = ((x1 + x2) / 2) / width
            cy = ((y1 + y2) / 2) / height
            bw = (x2 - x1) / width
            bh = (y2 - y1) / height
            if name == "price":
                price_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            elif name == "product":
                product_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        for out, lines, rows, tier in (
            (price_out, price_lines, price_rows, "price_tag_broad"),
            (product_out, product_lines, product_rows, "product_context"),
        ):
            if not lines:
                continue
            image_rel = Path("images") / image_path.name
            label_rel = Path("labels") / f"{image_path.stem}.txt"
            _link_or_copy(image_path, out / image_rel)
            _safe_mkdir((out / label_rel).parent)
            (out / label_rel).write_text("\n".join(lines) + "\n", encoding="utf-8")
            rows.append({
                "image": image_rel.as_posix(),
                "label": label_rel.as_posix(),
                "source_dataset": "hitl_supermarket_shelves",
                "source_image_id": image_path.stem,
                "tier": tier,
                "n_labels": len(lines),
                "license": "CC0-1.0",
                "source": DATASETS["hitl_supermarket_shelves"].source_url,
            })

    for out, rows, name in (
        (price_out, price_rows, "price_tag"),
        (product_out, product_rows, "product"),
    ):
        if not rows:
            continue
        _write_manifest(out / "manifest.csv", rows)
        image_list = [str((out / r["image"]).resolve()) for r in rows]
        (out / "train_images.txt").write_text("\n".join(image_list) + "\n", encoding="utf-8")
        (out / "val_images.txt").write_text("\n".join(image_list[: max(1, len(image_list) // 5)]) + "\n", encoding="utf-8")
        _write_yolo_dataset_yaml(out, [name])
        (out / "README.md").write_text(
            f"# HITL Supermarket Shelves {name} v1\n\n"
            f"Built from Pascal VOC XML. Images: {len(rows)}. "
            "External validation is not trusted; use real Lenta video holdout.\n",
            encoding="utf-8",
        )
        make_contact_sheet(out, out / "qa_contact_sheet.jpg", n=45)
    return price_out if price_rows else None, product_out if product_rows else None


def _hitl_json_image_path(json_path: Path) -> Path | None:
    name = json_path.name
    if name.endswith(".jpg.json"):
        image_name = name[:-5]
    elif name.endswith(".png.json"):
        image_name = name[:-5]
    else:
        image_name = json_path.with_suffix("").name
    for candidate in (
        json_path.parent / image_name,
        json_path.parent.parent / "images" / image_name,
        json_path.parent.parent.parent / "images" / image_name,
    ):
        if candidate.exists():
            return candidate
    matches = list(json_path.parent.parent.glob(f"**/{image_name}"))
    return matches[0] if matches else None


def build_hitl_supervisely_tasks(raw_dir: Path) -> tuple[Path | None, Path | None]:
    json_paths = sorted(p for p in raw_dir.glob("**/*.json") if p.name != "meta.json")
    if not json_paths:
        return None, None
    price_out = TASK_ROOT / "price_tag_detector" / "hitl_supermarket_shelves_price_tag_broad_v1"
    product_out = TASK_ROOT / "product_context" / "hitl_supermarket_shelves_products_v1"
    price_rows: list[dict[str, object]] = []
    product_rows: list[dict[str, object]] = []

    for json_path in json_paths:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        image_path = _hitl_json_image_path(json_path)
        if image_path is None:
            continue
        size = data.get("size", {})
        width = int(size.get("width") or Image.open(image_path).size[0])
        height = int(size.get("height") or Image.open(image_path).size[1])
        price_lines: list[str] = []
        product_lines: list[str] = []
        for obj in data.get("objects", []):
            if obj.get("geometryType") != "rectangle":
                continue
            title = str(obj.get("classTitle", "")).strip().lower()
            exterior = obj.get("points", {}).get("exterior", [])
            if len(exterior) != 2:
                continue
            (x1, y1), (x2, y2) = exterior
            x1, x2 = sorted((max(0.0, float(x1)), min(float(width), float(x2))))
            y1, y2 = sorted((max(0.0, float(y1)), min(float(height), float(y2))))
            if x2 <= x1 or y2 <= y1:
                continue
            cx = ((x1 + x2) / 2) / width
            cy = ((y1 + y2) / 2) / height
            bw = (x2 - x1) / width
            bh = (y2 - y1) / height
            if title == "price":
                price_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            elif title == "product":
                product_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        for out, lines, rows, tier in (
            (price_out, price_lines, price_rows, "price_tag_broad"),
            (product_out, product_lines, product_rows, "product_context"),
        ):
            if not lines:
                continue
            image_rel = Path("images") / image_path.name
            label_rel = Path("labels") / f"{image_path.stem}.txt"
            _link_or_copy(image_path, out / image_rel)
            _safe_mkdir((out / label_rel).parent)
            (out / label_rel).write_text("\n".join(lines) + "\n", encoding="utf-8")
            rows.append({
                "image": image_rel.as_posix(),
                "label": label_rel.as_posix(),
                "source_dataset": "hitl_supermarket_shelves",
                "source_image_id": image_path.stem,
                "tier": tier,
                "n_labels": len(lines),
                "license": "CC0-1.0",
                "source": DATASETS["hitl_supermarket_shelves"].source_url,
            })

    for out, rows, name in (
        (price_out, price_rows, "price_tag"),
        (product_out, product_rows, "product"),
    ):
        if not rows:
            continue
        _write_manifest(out / "manifest.csv", rows)
        image_list = [str((out / r["image"]).resolve()) for r in rows]
        (out / "train_images.txt").write_text("\n".join(image_list) + "\n", encoding="utf-8")
        (out / "val_images.txt").write_text("\n".join(image_list[: max(1, len(image_list) // 5)]) + "\n", encoding="utf-8")
        _write_yolo_dataset_yaml(out, [name])
        (out / "README.md").write_text(
            f"# HITL Supermarket Shelves {name} v1\n\n"
            f"Built from Supervisely JSON. Images: {len(rows)}. "
            "Price boxes are useful broad detector data; product boxes stay separate as context/hard-negative/product data.\n",
            encoding="utf-8",
        )
        make_contact_sheet(out, out / "qa_contact_sheet.jpg", n=45)
    return price_out if price_rows else None, product_out if product_rows else None


def _find_sku_root(raw_dir: Path) -> Path | None:
    for path in raw_dir.glob("**/annotations_train.csv"):
        return path.parent.parent if path.parent.name == "annotations" else path.parent
    return None


def build_sku110k_object_task(raw_dir: Path, max_images: int | None = None) -> Path | None:
    sku_root = _find_sku_root(raw_dir)
    if sku_root is None:
        return None
    out = TASK_ROOT / "object_pretrain" / "sku110k_object_v1"
    rows: list[dict[str, object]] = []
    for split, csv_name in (("train", "annotations_train.csv"), ("val", "annotations_val.csv"), ("test", "annotations_test.csv")):
        csv_path = sku_root / "annotations" / csv_name
        if not csv_path.exists():
            csv_path = sku_root / csv_name
        if not csv_path.exists():
            continue
        df = pd.read_csv(
            csv_path,
            header=None,
            names=["image", "x1", "y1", "x2", "y2", "class", "image_width", "image_height"],
        )
        if max_images:
            keep_images = sorted(df["image"].unique())[:max_images]
            df = df[df["image"].isin(keep_images)]
        for image_name, group in df.groupby("image"):
            image_path = sku_root / "images" / str(image_name)
            if not image_path.exists():
                continue
            lines = []
            for _, item in group.iterrows():
                width = float(item["image_width"])
                height = float(item["image_height"])
                x1, y1, x2, y2 = float(item["x1"]), float(item["y1"]), float(item["x2"]), float(item["y2"])
                cx = ((x1 + x2) / 2) / width
                cy = ((y1 + y2) / 2) / height
                bw = (x2 - x1) / width
                bh = (y2 - y1) / height
                if 0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1:
                    lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            if not lines:
                continue
            image_rel = Path("images") / split / image_path.name
            label_rel = Path("labels") / split / f"{image_path.stem}.txt"
            _link_or_copy(image_path, out / image_rel)
            _safe_mkdir((out / label_rel).parent)
            (out / label_rel).write_text("\n".join(lines) + "\n", encoding="utf-8")
            rows.append({
                "image": image_rel.as_posix(),
                "label": label_rel.as_posix(),
                "split": split,
                "source_dataset": "sku110k",
                "source_image_id": image_path.stem,
                "tier": "object_pretrain_not_price_tag",
                "n_labels": len(lines),
                "license": "academic/non-commercial; verify source terms",
                "source": DATASETS["sku110k"].source_url,
            })
    if not rows:
        return out
    _write_manifest(out / "manifest.csv", rows)
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"] or rows[: max(1, len(rows) // 10)]
    test_rows = [r for r in rows if r["split"] == "test"]
    (out / "train_images.txt").write_text("\n".join(str((out / r["image"]).resolve()) for r in train_rows) + "\n", encoding="utf-8")
    (out / "val_images.txt").write_text("\n".join(str((out / r["image"]).resolve()) for r in val_rows) + "\n", encoding="utf-8")
    (out / "test_images.txt").write_text("\n".join(str((out / r["image"]).resolve()) for r in test_rows) + ("\n" if test_rows else ""), encoding="utf-8")
    _write_yolo_dataset_yaml(out, ["object"], train="train_images.txt", val="val_images.txt")
    (out / "README.md").write_text(
        "# SKU-110K object pretrain v1\n\n"
        "This dataset is deliberately classed as `object`, not `price_tag`. It is useful for dense shelf/object "
        "pretraining and product/facing experiments, but mapping these boxes to price tags would corrupt the detector.\n\n"
        f"- Images: {len(rows)}\n",
        encoding="utf-8",
    )
    make_contact_sheet(out, out / "qa_contact_sheet.jpg", n=80)
    return out


def _read_manifest_rows(dataset_dir: Path) -> list[dict[str, object]]:
    manifest_path = dataset_dir / "manifest.csv"
    if not manifest_path.exists():
        return []
    rows: list[dict[str, object]] = []
    for row in csv.DictReader(manifest_path.open(encoding="utf-8")):
        out_row = dict(row)
        out_row["image"] = str((dataset_dir / str(row["image"])).resolve())
        out_row["label"] = str((dataset_dir / str(row["label"])).resolve()) if row.get("label") else ""
        out_row["component_dataset"] = dataset_dir.relative_to(TASK_ROOT).as_posix()
        rows.append(out_row)
    return rows


def _write_composite_dataset(out: Path, components: list[Path], names: list[str], title: str, notes: str) -> Path:
    rows: list[dict[str, object]] = []
    for component in components:
        rows.extend(_read_manifest_rows(component))
    rows = [row for row in rows if row.get("label")]
    _safe_mkdir(out)
    _write_manifest(out / "manifest.csv", rows)
    image_list = [str(row["image"]) for row in rows]
    # External composites are train/pretrain only. The val file is present for
    # loader smoke checks; real validation must use held-out Lenta video.
    (out / "train_images.txt").write_text("\n".join(image_list) + ("\n" if image_list else ""), encoding="utf-8")
    (out / "val_images.txt").write_text(
        "\n".join(image_list[: max(1, len(image_list) // 20)]) + ("\n" if image_list else ""),
        encoding="utf-8",
    )
    _write_yolo_dataset_yaml(out, names)
    boxes = sum(int(float(row.get("n_labels") or 0)) for row in rows)
    (out / "README.md").write_text(
        f"# {title}\n\n"
        f"{notes}\n\n"
        f"- Images: {len(rows)}\n"
        f"- Boxes: {boxes}\n"
        "- This composite is for training/pretraining convenience only. Do not use its `val_images.txt` as a final metric.\n\n"
        "## Components\n\n"
        + "\n".join(f"- `{component.relative_to(TASK_ROOT).as_posix()}`" for component in components)
        + "\n",
        encoding="utf-8",
    )
    if rows:
        make_contact_sheet(out, out / "qa_contact_sheet.jpg", n=100)
    return out


def build_composites() -> list[Path]:
    price_clean = _write_composite_dataset(
        TASK_ROOT / "composites" / "external_price_tag_clean_train_v1",
        [
            TASK_ROOT / "price_tag_detector" / "openfoodfacts_lenta_strict_candidates_v1",
            TASK_ROOT / "price_tag_detector" / "sovar_roboflow_price_tag_clean_v1",
            TASK_ROOT / "price_tag_detector" / "hitl_supermarket_shelves_price_tag_broad_v1",
        ],
        ["price_tag"],
        "External price-tag clean train v1",
        "Cleaner external detector-training mix: OpenFoodFacts strict candidates, filtered SOVAR, and HITL Price boxes.",
    )
    price_broad = _write_composite_dataset(
        TASK_ROOT / "composites" / "external_price_tag_broad_train_v1",
        [
            TASK_ROOT / "price_tag_detector" / "openfoodfacts_lenta_strict_candidates_v1",
            TASK_ROOT / "price_tag_detector" / "openfoodfacts_price_tag_broad_v1",
            TASK_ROOT / "price_tag_detector" / "sovar_roboflow_price_tag_clean_v1",
            TASK_ROOT / "price_tag_detector" / "hitl_supermarket_shelves_price_tag_broad_v1",
        ],
        ["price_tag"],
        "External price-tag broad train v1",
        "Broader external price-tag detector mix. Uses filtered SOVAR rather than raw SOVAR broad to avoid redaction-box poisoning.",
    )
    object_pretrain = _write_composite_dataset(
        TASK_ROOT / "composites" / "external_object_pretrain_v1",
        [
            TASK_ROOT / "object_pretrain" / "sku110k_object_v1",
            TASK_ROOT / "product_context" / "hitl_supermarket_shelves_products_v1",
        ],
        ["object"],
        "External object pretrain v1",
        "Dense retail object/product pretraining mix. This is not a price-tag dataset.",
    )
    return [price_clean, price_broad, object_pretrain]


def build_available() -> list[str]:
    outputs: list[str] = []
    outputs.extend(str(p) for p in build_openfoodfacts_tasks())
    rf_out = build_roboflow_yolo_tasks(
        "sovar_roboflow",
        DATASETS["sovar_roboflow"].raw_dir,
        DATASETS["sovar_roboflow"].source_url,
    )
    if rf_out:
        outputs.append(str(rf_out))
    sovar_clean = build_sovar_clean_task()
    if sovar_clean:
        outputs.append(str(sovar_clean))
    hitl_outs = build_hitl_supervisely_tasks(DATASETS["hitl_supermarket_shelves"].raw_dir)
    if not any(hitl_outs):
        hitl_outs = build_hitl_voc_tasks(DATASETS["hitl_supermarket_shelves"].raw_dir)
    outputs.extend(str(p) for p in hitl_outs if p)
    sku_out = build_sku110k_object_task(DATASETS["sku110k"].raw_dir)
    if sku_out:
        outputs.append(str(sku_out))
    outputs.extend(str(path) for path in build_composites())
    index_path = TASK_ROOT / "README.md"
    _safe_mkdir(TASK_ROOT)
    index_path.write_text(
        "# Task-specific external research datasets\n\n"
        "Generated outputs are grouped by model/training purpose. These sets remain under `dataset_research/` "
        "until manually audited; final validation must stay on held-out real Lenta video.\n\n"
        + "\n".join(f"- `{Path(path).relative_to(TASK_ROOT).as_posix()}`" for path in outputs)
        + ("\n" if outputs else ""),
        encoding="utf-8",
    )
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status", help="Write dataset_research/external_dataset_status.json")
    sub.add_parser("download-runbook", help="Write dataset_research/external_dataset_download_runbook.md")
    sub.add_parser("build-available", help="Build task datasets for raw data currently present locally")
    sheet = sub.add_parser("contact-sheet", help="Build one contact sheet from an output task dataset")
    sheet.add_argument("dataset_dir", type=Path)
    sheet.add_argument("--out", type=Path, default=None)
    sheet.add_argument("--n", type=int, default=80)
    args = parser.parse_args()

    if args.cmd == "status":
        print(json.dumps(status(), indent=2, ensure_ascii=False))
    elif args.cmd == "download-runbook":
        print(write_download_runbook())
    elif args.cmd == "build-available":
        print(json.dumps(build_available(), indent=2, ensure_ascii=False))
    elif args.cmd == "contact-sheet":
        out = args.out or args.dataset_dir / "qa_contact_sheet.jpg"
        make_contact_sheet(args.dataset_dir, out, n=args.n)
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
