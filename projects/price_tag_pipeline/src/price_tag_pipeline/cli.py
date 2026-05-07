from __future__ import annotations

import argparse
import logging

from .config import load_config
from .pipeline import PriceTagPipeline


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Price tag video pipeline runner")
    p.add_argument("--video", required=True, help="Absolute path to input video")
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--output", default=None, help="Optional JSONL output path")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    cfg = load_config(args.config)
    pipe = PriceTagPipeline(cfg)
    preds = pipe.run(video_path=args.video, output_jsonl=args.output)
    logging.getLogger(__name__).info("Done. Final predictions: %d", len(preds))

