#!/usr/bin/env python3
"""End-to-end inference: video -> structured JSON tags.

Usage:
    python projects/price_tag_pipeline/scripts/run_inference.py \\
        --video path/to/video.mp4 \\
        --config projects/price_tag_pipeline/configs/balanced.yaml \\
        --output outputs/video01.jsonl
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

from price_tag_pipeline.cli import main  # noqa: E402


if __name__ == "__main__":
    main()
