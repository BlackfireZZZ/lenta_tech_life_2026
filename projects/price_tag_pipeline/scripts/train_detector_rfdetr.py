#!/usr/bin/env python3
"""Train an RF-DETR detector (Roboflow, ICLR 2026).

Status: STUB. The full training loop lands when:
    1. The dataset is in COCO format (RF-DETR's native format), OR
       a YOLO->COCO converter is hooked in. We already have COCO ingest in
       price_tag_pipeline.data.loaders so this is mostly a path question.
    2. `rfdetr` is installed: pip install rfdetr

Usage (once implemented):
    python projects/price_tag_pipeline/scripts/train_detector_rfdetr.py \\
        --dataset data/processed/coco/ \\
        --variant base \\
        --epochs 60 \\
        --batch 8 \\
        --device 0

References:
    https://github.com/roboflow/rf-detr
    docs/strategy.md §2.1
"""

from __future__ import annotations

import argparse
import logging


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--variant", default="base", choices=["nano", "small", "base", "large", "2xl"])
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--device", default="0")
    p.add_argument("--lr-backbone", type=float, default=1e-4)
    p.add_argument("--lr-head", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--project", default="runs/lenta")
    p.add_argument("--name", default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO)

    try:
        import rfdetr  # type: ignore  # noqa: F401
    except ImportError:
        raise SystemExit(
            "rfdetr is not installed yet. Install with: pip install rfdetr\n"
            "Then implement this script per https://github.com/roboflow/rf-detr"
        )

    raise NotImplementedError(
        "RF-DETR training body is not implemented yet. "
        "Hook it up here once data is in COCO format. See docs/strategy.md §2.1."
    )


if __name__ == "__main__":
    raise SystemExit(main())
