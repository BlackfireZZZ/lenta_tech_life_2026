"""Command-line entry point for end-to-end video inference."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import load_config
from .pipeline import PriceTagPipeline


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Price tag video pipeline runner")
    p.add_argument("--video", required=True, help="Absolute path to input video")
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--output", default=None,
                   help="Optional output path. .json => JSON array, anything else => JSONL.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    cfg = load_config(args.config)
    pipe = PriceTagPipeline(cfg)
    tags = pipe.run(video_path=str(video_path), output_path=args.output)
    logging.getLogger(__name__).info("Done. Final tags: %d", len(tags))


if __name__ == "__main__":
    main()
