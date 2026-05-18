#!/usr/bin/env python3
"""Build a compact Colab-ready product_facing pretrain bundle.

This intentionally uses only product/object datasets. Price-tag datasets are
excluded because they are positives for the OCR/tag detector, not for
product_facing.

Output layout:
    product_facing_pretrain_colab_bundle/
      retail_product_facing_pretrain_yolo/
        images/{train,val,test}/
        labels/{train,val,test}/
        data.yaml
      product_facing_pretrain_colab.ipynb
      README.md
      manifests/build_manifest.csv
      reports/dataset_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

DEFAULT_RESEARCH_ROOT = Path("/Users/cute/Lenta_Tech/dataset_research")
DEFAULT_OUT = Path("/Users/cute/Lenta_Tech/product_facing_pretrain_colab_bundle")
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class SourceItem:
    source: str
    split: str
    image: Path
    label: Path
    source_id: str


@dataclass(frozen=True)
class CleanedLabels:
    lines: tuple[str, ...]
    dropped: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--research-root", type=Path, default=DEFAULT_RESEARCH_ROOT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--sku-train", type=int, default=1800)
    p.add_argument("--sku-val", type=int, default=250)
    p.add_argument("--sku-test", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-box-area", type=float, default=0.000015)
    p.add_argument("--max-box-area", type=float, default=0.45)
    p.add_argument("--min-box-side", type=float, default=0.0015)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    research_root = args.research_root.expanduser().resolve()
    out_root = args.out.expanduser().resolve()
    dataset_dir = out_root / "retail_product_facing_pretrain_yolo"

    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)
    if out_root.exists() and any(out_root.iterdir()) and not args.overwrite:
        raise SystemExit(f"{out_root} already exists. Re-run with --overwrite.")

    _make_dirs(dataset_dir)
    (out_root / "manifests").mkdir(parents=True, exist_ok=True)
    (out_root / "reports").mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    sku_items = _load_sku110k_items(research_root)
    hitl_items = _load_hitl_product_items(research_root)

    selected: list[SourceItem] = []
    selected.extend(_sample_split(sku_items, "train", args.sku_train, rng))
    selected.extend(_sample_split(sku_items, "val", args.sku_val, rng))
    selected.extend(_sample_split(sku_items, "test", args.sku_test, rng))
    selected.extend(hitl_items)

    manifest_rows: list[dict[str, object]] = []
    summary = {
        "dataset_name": "retail_product_facing_pretrain_yolo",
        "class_names": ["product_facing"],
        "seed": args.seed,
        "sources": {},
        "splits": {},
        "cleaning": {
            "min_box_area": args.min_box_area,
            "max_box_area": args.max_box_area,
            "min_box_side": args.min_box_side,
            "all_classes_remapped_to": "0 product_facing",
            "price_tag_datasets_included": False,
        },
    }

    split_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    box_counts: Counter[str] = Counter()
    dropped_counts: Counter[str] = Counter()

    for item in selected:
        cleaned = _clean_yolo_label(
            item.label,
            min_area=args.min_box_area,
            max_area=args.max_box_area,
            min_side=args.min_box_side,
        )
        if not cleaned.lines:
            dropped_counts[f"{item.source}:empty_after_clean"] += 1
            continue

        ext = item.image.suffix.lower() if item.image.suffix else ".jpg"
        stem = f"{item.source}_{item.source_id}".replace("/", "_").replace(" ", "_")
        dst_img = dataset_dir / "images" / item.split / f"{stem}{ext}"
        dst_label = dataset_dir / "labels" / item.split / f"{stem}.txt"

        shutil.copy2(item.image, dst_img, follow_symlinks=True)
        dst_label.write_text("\n".join(cleaned.lines) + "\n", encoding="utf-8")

        split_counts[item.split] += 1
        source_counts[item.source] += 1
        box_counts[item.split] += len(cleaned.lines)
        dropped_counts[f"{item.source}:boxes_dropped"] += cleaned.dropped
        manifest_rows.append(
            {
                "split": item.split,
                "source": item.source,
                "source_id": item.source_id,
                "image": str(dst_img.relative_to(out_root)),
                "label": str(dst_label.relative_to(out_root)),
                "source_image": str(item.image),
                "source_label": str(item.label),
                "n_labels": len(cleaned.lines),
                "n_labels_dropped": cleaned.dropped,
            }
        )

    _write_data_yaml(dataset_dir)
    _write_manifest(out_root / "manifests" / "build_manifest.csv", manifest_rows)
    (out_root / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), out_root / "scripts" / Path(__file__).name)

    summary["splits"] = {
        split: {
            "images": split_counts[split],
            "boxes": box_counts[split],
        }
        for split in SPLITS
    }
    summary["sources"] = dict(source_counts)
    summary["dropped"] = dict(dropped_counts)
    (out_root / "reports" / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    _write_readme(out_root, summary)
    _write_colab_notebook(out_root / "product_facing_pretrain_colab.ipynb")

    print(f"Built bundle: {out_root}")
    print(json.dumps(summary["splits"], ensure_ascii=False, indent=2))
    return 0


def _make_dirs(dataset_dir: Path) -> None:
    for split in SPLITS:
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def _load_sku110k_items(research_root: Path) -> list[SourceItem]:
    root = research_root / "task_datasets" / "object_pretrain" / "sku110k_object_v1"
    items: list[SourceItem] = []
    for split in SPLITS:
        labels_dir = root / "labels" / split
        images_dir = root / "images" / split
        for label in sorted(labels_dir.glob("*.txt")):
            image = images_dir / f"{label.stem}.jpg"
            if image.exists():
                items.append(SourceItem("sku110k", split, image, label, label.stem))
    return items


def _load_hitl_product_items(research_root: Path) -> list[SourceItem]:
    root = research_root / "task_datasets" / "product_context" / "hitl_supermarket_shelves_products_v1"
    items: list[SourceItem] = []
    val_images = _read_image_list(root / "val_images.txt")
    val_stems = {p.stem for p in val_images}
    for split in ("train", "val"):
        for image in _read_image_list(root / f"{split}_images.txt"):
            if split == "train" and image.stem in val_stems:
                continue
            label = root / "labels" / f"{image.stem}.txt"
            if image.exists() and label.exists():
                items.append(SourceItem("hitl_products", split, image, label, image.stem))
    return items


def _read_image_list(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return [Path(raw.strip()) for raw in path.read_text(encoding="utf-8").splitlines() if raw.strip()]


def _sample_split(items: list[SourceItem], split: str, limit: int, rng: random.Random) -> list[SourceItem]:
    pool = [item for item in items if item.split == split]
    if limit <= 0 or limit >= len(pool):
        return pool
    pool = list(pool)
    rng.shuffle(pool)
    return sorted(pool[:limit], key=lambda x: x.source_id)


def _clean_yolo_label(path: Path, min_area: float, max_area: float, min_side: float) -> CleanedLabels:
    lines: list[str] = []
    dropped = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if len(parts) < 5:
            dropped += 1
            continue
        try:
            cx, cy, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            dropped += 1
            continue
        area = w * h
        x1, x2 = cx - w / 2.0, cx + w / 2.0
        y1, y2 = cy - h / 2.0, cy + h / 2.0
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
            dropped += 1
            continue
        if w < min_side or h < min_side or area < min_area or area > max_area:
            dropped += 1
            continue
        if x1 < -0.002 or x2 > 1.002 or y1 < -0.002 or y2 > 1.002:
            dropped += 1
            continue
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return CleanedLabels(tuple(lines), dropped)


def _write_data_yaml(dataset_dir: Path) -> None:
    dataset_dir.joinpath("data.yaml").write_text(
        "\n".join(
            [
                "path: .",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "nc: 1",
                "names:",
                "  0: product_facing",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "split",
        "source",
        "source_id",
        "image",
        "label",
        "source_image",
        "source_label",
        "n_labels",
        "n_labels_dropped",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_readme(out_root: Path, summary: dict) -> None:
    out_root.joinpath("README.md").write_text(
        f"""# Product Facing Pretrain Colab Bundle

Compact YOLO dataset for first-stage `product_facing` detector pretraining.

Upload this whole folder to Google Drive:

```text
MyDrive/Lenta/product_facing_pretrain_colab_bundle/
```

Then open:

```text
product_facing_pretrain_colab.ipynb
```

Dataset:

```text
retail_product_facing_pretrain_yolo/
  images/train|val|test
  labels/train|val|test
  data.yaml
```

Class schema:

```yaml
0: product_facing
```

Included sources:

- SKU-110K object/facing subset.
- HITL Supermarket Shelves product-context images.

Excluded on purpose:

- OpenFoodFacts/SOVAR/other price-tag datasets as positive samples.

Split summary:

```json
{json.dumps(summary["splits"], ensure_ascii=False, indent=2)}
```

Use this as broad retail pretraining, then fine-tune on reviewed Lenta
keyframes.
""",
        encoding="utf-8",
    )


def _write_colab_notebook(path: Path) -> None:
    notebook = {
        "cells": [
            _md("# Product Facing YOLO Pretrain\n\nTrain one-class `product_facing` detector on the prepared retail pretrain bundle, then save weights back to Google Drive."),
            _code(
                "from google.colab import drive\n"
                "drive.mount('/content/drive')\n"
            ),
            _code(
                "from pathlib import Path\n\n"
                "DRIVE_ROOT = Path('/content/drive/MyDrive/Lenta/product_facing_pretrain_colab_bundle')\n"
                "DATASET_DIR = DRIVE_ROOT / 'retail_product_facing_pretrain_yolo'\n"
                "DATA_YAML = DATASET_DIR / 'data.yaml'\n"
                "RUNS_DIR = DRIVE_ROOT / 'runs'\n"
                "MODELS_DIR = DRIVE_ROOT / 'models'\n"
                "REPORTS_DIR = DRIVE_ROOT / 'reports'\n"
                "RUNS_DIR.mkdir(parents=True, exist_ok=True)\n"
                "MODELS_DIR.mkdir(parents=True, exist_ok=True)\n"
                "REPORTS_DIR.mkdir(parents=True, exist_ok=True)\n\n"
                "assert DATA_YAML.exists(), f'Missing dataset yaml: {DATA_YAML}'\n"
                "DATA_YAML.write_text(\n"
                "    f'path: {DATASET_DIR}\\n'\n"
                "    'train: images/train\\n'\n"
                "    'val: images/val\\n'\n"
                "    'test: images/test\\n'\n"
                "    'nc: 1\\n'\n"
                "    'names:\\n'\n"
                "    '  0: product_facing\\n'\n"
                ")\n"
                "print(DATA_YAML.read_text())\n"
            ),
            _code(
                "!pip -q install -U ultralytics\n"
            ),
            _code(
                "from pathlib import Path\n"
                "import json\n\n"
                "def count_split(split):\n"
                "    images = list((DATASET_DIR / 'images' / split).glob('*'))\n"
                "    labels = list((DATASET_DIR / 'labels' / split).glob('*.txt'))\n"
                "    boxes = 0\n"
                "    for label in labels:\n"
                "        boxes += sum(1 for line in label.read_text().splitlines() if line.strip())\n"
                "    return {'images': len(images), 'labels': len(labels), 'boxes': boxes}\n\n"
                "summary = {split: count_split(split) for split in ['train', 'val', 'test']}\n"
                "print(json.dumps(summary, indent=2, ensure_ascii=False))\n"
            ),
            _code(
                "from ultralytics import YOLO\n"
                "import shutil\n\n"
                "BASE_MODEL = 'yolo11s.pt'  # yolo11n.pt for faster smoke, yolo11m.pt if GPU allows\n"
                "RUN_NAME = 'product_facing_retail_pretrain_yolo11s'\n\n"
                "model = YOLO(BASE_MODEL)\n"
                "results = model.train(\n"
                "    data=str(DATA_YAML),\n"
                "    epochs=80,\n"
                "    imgsz=1280,\n"
                "    batch=-1,\n"
                "    device=0,\n"
                "    workers=8,\n"
                "    patience=20,\n"
                "    project=str(RUNS_DIR),\n"
                "    name=RUN_NAME,\n"
                "    optimizer='auto',\n"
                "    cos_lr=True,\n"
                "    amp=True,\n"
                "    cache='disk',\n"
                "    seed=42,\n"
                "    close_mosaic=10,\n"
                "    mosaic=1.0,\n"
                "    mixup=0.05,\n"
                "    copy_paste=0.0,\n"
                "    hsv_h=0.015,\n"
                "    hsv_s=0.55,\n"
                "    hsv_v=0.35,\n"
                "    degrees=2.0,\n"
                "    translate=0.08,\n"
                "    scale=0.35,\n"
                "    shear=1.0,\n"
                "    fliplr=0.5,\n"
                "    plots=True,\n"
                ")\n\n"
                "run_dir = RUNS_DIR / RUN_NAME\n"
                "best = run_dir / 'weights' / 'best.pt'\n"
                "last = run_dir / 'weights' / 'last.pt'\n"
                "assert best.exists(), f'Missing best checkpoint: {best}'\n"
                "shutil.copy2(best, MODELS_DIR / 'product_facing_retail_pretrain_yolo11s_best.pt')\n"
                "shutil.copy2(last, MODELS_DIR / 'product_facing_retail_pretrain_yolo11s_last.pt')\n"
                "print('Saved:', MODELS_DIR)\n"
            ),
            _code(
                "from ultralytics import YOLO\n\n"
                "best_model = YOLO(str(MODELS_DIR / 'product_facing_retail_pretrain_yolo11s_best.pt'))\n"
                "metrics = best_model.val(data=str(DATA_YAML), split='test', imgsz=1280, device=0, plots=True)\n"
                "print(metrics)\n"
            ),
            _md(
                "## Fine-tune on reviewed Lenta keyframes later\n\n"
                "After CVAT cleanup, upload a target-domain YOLO dataset to `MyDrive/Lenta/lenta_product_facing_reviewed_v1/` and start from `models/product_facing_retail_pretrain_yolo11s_best.pt`."
            ),
            _code(
                "# Optional target-domain fine-tune template. Run only after reviewed Lenta labels exist.\n"
                "from pathlib import Path\n\n"
                "LENTA_DATA = Path('/content/drive/MyDrive/Lenta/lenta_product_facing_reviewed_v1/data.yaml')\n"
                "if LENTA_DATA.exists():\n"
                "    model = YOLO(str(MODELS_DIR / 'product_facing_retail_pretrain_yolo11s_best.pt'))\n"
                "    model.train(\n"
                "        data=str(LENTA_DATA),\n"
                "        epochs=60,\n"
                "        imgsz=1280,\n"
                "        batch=-1,\n"
                "        device=0,\n"
                "        workers=8,\n"
                "        patience=15,\n"
                "        project=str(RUNS_DIR),\n"
                "        name='product_facing_lenta_finetune_v1',\n"
                "        optimizer='auto',\n"
                "        cos_lr=True,\n"
                "        amp=True,\n"
                "        close_mosaic=10,\n"
                "        seed=42,\n"
                "        plots=True,\n"
                "    )\n"
                "else:\n"
                "    print('Skip: reviewed Lenta dataset not found yet:', LENTA_DATA)\n"
            ),
        ],
        "metadata": {
            "accelerator": "GPU",
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def _code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


if __name__ == "__main__":
    raise SystemExit(main())
