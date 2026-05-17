#!/usr/bin/env python3
"""Run inference for every video in a directory.

Usage:
    python projects/price_tag_pipeline/scripts/run_batch_inference.py \
        --videos-dir data/raw/videos \
        --config projects/price_tag_pipeline/configs/balanced.yaml \
        --outputs-dir outputs/jsonl
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
SRC = THIS.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from price_tag_pipeline.config import load_config  # noqa: E402


def _collect_videos(videos_dir: Path, pattern: str) -> list[Path]:
    videos = sorted(videos_dir.glob(pattern))
    return [p for p in videos if p.is_file()]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--videos-dir", required=True, help="Directory with input videos")
    p.add_argument("--config", required=True, help="Pipeline YAML config path")
    p.add_argument("--outputs-dir", required=True, help="Where per-video JSONL files are written")
    p.add_argument("--pattern", default="*.mp4", help="Glob for input videos")
    p.add_argument("--limit", type=int, default=0, help="Optional limit for debugging")
    p.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True,
                   help="Per-video terminal progress bar (degrades to log lines without tqdm).")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    videos_dir = Path(args.videos_dir).expanduser().resolve()
    outputs_dir = Path(args.outputs_dir).expanduser().resolve()
    outputs_dir.mkdir(parents=True, exist_ok=True)
    if not videos_dir.exists():
        raise SystemExit(f"Videos directory not found: {videos_dir}")

    videos = _collect_videos(videos_dir, args.pattern)
    if args.limit > 0:
        videos = videos[: args.limit]
    if not videos:
        raise SystemExit(f"No videos found at {videos_dir} with pattern '{args.pattern}'")

    # Import late so `--help` works even when heavy runtime deps are missing.
    from price_tag_pipeline.pipeline import PriceTagPipeline  # noqa: E402
    from price_tag_pipeline.progress import TqdmProgress  # noqa: E402

    cfg = load_config(args.config)
    pipe = PriceTagPipeline(cfg)

    logging.info("Found %d video(s). Starting batch inference...", len(videos))
    for idx, video in enumerate(videos, start=1):
        out_path = outputs_dir / f"{video.stem}.jsonl"
        logging.info("[%d/%d] %s -> %s", idx, len(videos), video.name, out_path.name)
        # A fresh bar per video; the pipeline closes it in its own finally.
        tags = pipe.run(
            video_path=str(video),
            output_path=str(out_path),
            progress=TqdmProgress() if args.progress else None,
        )
        logging.info("[%d/%d] done: %s tags=%d", idx, len(videos), video.name, len(tags))

    logging.info("Batch inference finished. Outputs at: %s", outputs_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
