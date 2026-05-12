#!/usr/bin/env python3
"""Stage raw data into a normalized, validated, train-ready format.

Steps:
1. Detect annotation format (YOLO / COCO).
2. Extract frames from videos if needed.
3. Mirror labels into processed/labels/{video_id}/.
4. Run integrity checks, abort on failure unless --allow-bad.
5. Emit a default dataset.yaml (no split yet — run make_splits.py for that).

Idempotent: re-running skips frames that are already extracted.

Usage:
    python projects/price_tag_pipeline/scripts/prepare_data.py \\
        --raw data/raw \\
        --processed data/processed
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

from price_tag_pipeline.data.loaders import (  # noqa: E402
    detect_format,
    ingest_yolo,
    coco_to_yolo,
    write_dataset_yaml,
)
from price_tag_pipeline.data.validate import validate_dataset  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Stage and validate raw data")
    p.add_argument("--raw", default="data/raw", help="Path to data/raw")
    p.add_argument("--processed", default="data/processed", help="Path to data/processed")
    p.add_argument("--no-extract", action="store_true", help="Do not extract frames from video files")
    p.add_argument("--allow-bad", action="store_true", help="Do not fail on integrity errors (just warn)")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    raw = Path(args.raw).resolve()
    processed = Path(args.processed).resolve()
    processed.mkdir(parents=True, exist_ok=True)

    fmt = detect_format(raw)
    logging.info("Detected raw format: %s", fmt)

    if fmt == "empty":
        logging.error(
            "No annotations found under %s. Drop YOLO labels (annotations/labels/<video>/*.txt) "
            "or a COCO json (annotations/*.json) and re-run.",
            raw,
        )
        return 1

    if fmt == "yolo":
        _, classes = ingest_yolo(raw, processed, extract_videos=not args.no_extract)
    elif fmt == "coco":
        _, classes = coco_to_yolo(raw, processed)
    elif fmt == "frames_only":
        logging.error("Frames found but no annotations. Drop labels under raw/annotations/labels/ first.")
        return 1
    else:
        logging.error("Unsupported format: %s", fmt)
        return 1

    report = validate_dataset(processed, classes, read_images=False)
    print("\nIntegrity report:")
    print(report.summary())
    print()
    if not report.is_ok and not args.allow_bad:
        logging.error("Integrity errors detected. Fix the data or re-run with --allow-bad.")
        return 2

    yaml_path = write_dataset_yaml(processed, classes, fold=None)
    logging.info("Wrote %s (no split — run make_splits.py next)", yaml_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
