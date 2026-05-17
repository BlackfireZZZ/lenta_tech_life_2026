#!/usr/bin/env python3
"""LoRA fine-tune PaddleOCR-VL 1.5 (0.9B) on annotated price-tag crops.

Status: STUB. Lands once we have:
    1. Annotated (crop_image, expected_json) pairs (≥1000 recommended).
    2. paddleocr>=3 OR transformers + peft installed.

Strategy reference: docs/strategy.md §3.3.

Expected dataset layout for fine-tuning:
    data/processed/vlm/
      train.jsonl        # one line per sample, fields: image_path, target_json
      val.jsonl

Each line:
    {"image": "path/to/crop.jpg", "target": "{\\"regular_price\\": ...}"}

Usage (once implemented):
    python projects/price_tag_pipeline/scripts/train_vlm_lora.py \\
        --train data/processed/vlm/train.jsonl \\
        --val   data/processed/vlm/val.jsonl \\
        --output data/checkpoints/vlm/paddle_vl_lora_run1 \\
        --epochs 3 --lr 5e-5 --r 16 --alpha 32 --device cuda:0
"""

from __future__ import annotations

import argparse


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--r", type=int, default=16, help="LoRA rank")
    p.add_argument("--alpha", type=int, default=32, help="LoRA alpha")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    raise NotImplementedError(
        "VLM LoRA training body is not implemented yet. "
        "Implement once annotated (crop, target_json) pairs exist. "
        "See docs/strategy.md §3.3 for the recipe."
    )


if __name__ == "__main__":
    raise SystemExit(main())
