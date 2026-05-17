#!/usr/bin/env python3
"""Train a YOLO-family detector via Ultralytics.

Defaults to fine-tuning the OpenFoodFacts price-tag detector on our
1280-shortest-side input. Switch with --model.

Usage:
    python projects/price_tag_pipeline/scripts/train_detector_yolo.py \\
        --dataset data/processed/dataset.yaml \\
        --model hf://openfoodfacts/price-tag-detection/weights/best.pt \\
        --epochs 200 \\
        --imgsz 1280 \\
        --batch 8 \\
        --device 0 \\
        --name yolo26l_fold0

For a different fold, first re-emit dataset.yaml:
    python projects/price_tag_pipeline/scripts/make_splits.py --emit-dataset-yaml-fold 1
then re-run this script.

To override augmentation, edit price_tag_pipeline/training/augmentation.py
and re-run. To plug in Albumentations transforms (motion blur, etc), pass
--use-albu and ensure albumentations is installed.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
SRC = THIS.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from price_tag_pipeline.training.augmentation import UltralyticsAugConfig  # noqa: E402
from price_tag_pipeline.detector import resolve_detector_model_path  # noqa: E402


DEFAULT_DETECTOR_MODEL = "hf://openfoodfacts/price-tag-detection/weights/best.pt"


def _default_run_name(model_arg: str, seed: int) -> str:
    if model_arg.startswith("hf://openfoodfacts/price-tag-detection/"):
        return f"openfoodfacts_price_tag_detection_seed{seed}"
    return f"{Path(model_arg).stem}_seed{seed}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="Path to dataset.yaml")
    p.add_argument("--model", default=DEFAULT_DETECTOR_MODEL,
                   help="Pretrained checkpoint, hf:// URI, or Ultralytics model name")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr0", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--project", default="runs/lenta")
    p.add_argument("--name", default=None)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--optimizer", default="auto", choices=["auto", "SGD", "AdamW"])
    p.add_argument("--wandb", action="store_true", help="Enable W&B logging via Ultralytics integration")
    p.add_argument("--use-albu", action="store_true",
                   help="Use the Albumentations integration in price_tag_pipeline.training.augmentation_albu")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.wandb:
        os.environ["WANDB_PROJECT"] = os.environ.get("WANDB_PROJECT", "lenta-2026")

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit(
            "ultralytics is not installed. pip install ultralytics"
        ) from e

    name = args.name or _default_run_name(args.model, args.seed)
    aug = UltralyticsAugConfig()
    model_path = resolve_detector_model_path(args.model)

    logging.info("Starting training: model=%s data=%s epochs=%d imgsz=%d batch=%d",
                 args.model, args.dataset, args.epochs, args.imgsz, args.batch)

    model = YOLO(model_path)
    train_kwargs = dict(
        data=args.dataset,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        lr0=args.lr0,
        seed=args.seed,
        project=args.project,
        name=name,
        patience=args.patience,
        optimizer=args.optimizer,
        verbose=True,
        amp=True,
        cos_lr=True,
        plots=True,
        save=True,
        **aug.as_kwargs(),
    )

    if args.use_albu:
        from price_tag_pipeline.training.augmentation_albu import attach_to_ultralytics
        attached = attach_to_ultralytics(model)
        logging.info("Heavy Albumentations attached: %s", attached)

    results = model.train(**train_kwargs)
    logging.info("Done. Best checkpoint: %s", model.trainer.best if hasattr(model, "trainer") else "see runs/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
