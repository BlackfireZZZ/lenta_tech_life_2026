#!/usr/bin/env python3
"""Assemble per-video JSONLs into a single submission artifact (JSON + CSV).

Usage:
    python projects/price_tag_pipeline/scripts/build_submission.py \\
        --inputs outputs \\
        --out-json submission/submission.json \\
        --out-csv  submission/submission.csv \\
        --strict
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

from price_tag_pipeline.submission import build_submission  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", required=True, help="Directory with per-video *.jsonl files")
    p.add_argument("--out-json", default=None, help="Path to write the JSON submission")
    p.add_argument("--out-csv", default=None, help="Path to write the CSV submission")
    p.add_argument("--strict", action="store_true", help="Fail on the first schema-invalid tag")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    build_submission(
        inputs_dir=Path(args.inputs),
        out_json=Path(args.out_json) if args.out_json else None,
        out_csv=Path(args.out_csv) if args.out_csv else None,
        strict=args.strict,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
