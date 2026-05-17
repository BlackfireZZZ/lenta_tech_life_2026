#!/usr/bin/env python3
"""Export per-video JSONL predictions to the graded hackathon CSV schema.

Input:
  Directory with *.jsonl files produced by run_inference / run_batch_inference.

Output:
  One CSV where each row is one predicted price tag (schema: docs/hackathon/
  task.md §3 — 29 columns).

The 29-column mapping lives in ``price_tag_pipeline.submission`` (the single
source of truth, also used by the ML service). This script is just the
JSONL-replay CLI front-end:

  - Fields unsupported by the current baseline are emitted as empty strings.
  - `frame_timestamp` is emitted in milliseconds.
  - `filename` defaults to the JSONL stem because the released Lenta CSVs use
    stems such as `25_2-10`, not `25_2-10.mp4`.
  - `price_discount` is derived as max(price_default - price_card, 0) when both
    prices are present; otherwise empty.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

THIS = Path(__file__).resolve().parent
SRC = THIS.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from price_tag_pipeline.submission import (  # noqa: E402
    HACK_CSV_COLUMNS,
    hack_row_from_tag_dict,
    hack_rows_to_csv_text,
)

# Back-compat alias: external callers / tests may still import this name.
CSV_COLUMNS = HACK_CSV_COLUMNS


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", required=True, help="Directory with per-video *.jsonl files")
    p.add_argument("--out-csv", required=True, help="Output CSV path")
    p.add_argument(
        "--video-ext",
        default="",
        help="Optional video extension appended to JSONL stem for `filename` (default: none)",
    )
    args = p.parse_args()

    inputs_dir = Path(args.inputs).expanduser().resolve()
    out_csv = Path(args.out_csv).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    ext = ""
    if args.video_ext:
        ext = args.video_ext if args.video_ext.startswith(".") else f".{args.video_ext}"

    jsonl_files = [p for p in sorted(inputs_dir.glob("*.jsonl")) if not p.name.endswith("_audit.jsonl")]
    if not jsonl_files:
        raise SystemExit(f"No JSONL files found in: {inputs_dir}")

    rows_out: list[dict[str, str]] = []
    for jf in jsonl_files:
        video_filename = f"{jf.stem}{ext}"
        for tag in _read_jsonl(jf):
            rows_out.append(hack_row_from_tag_dict(tag, filename=video_filename))

    out_csv.write_text(hack_rows_to_csv_text(rows_out), encoding="utf-8", newline="")
    print(f"Wrote {len(rows_out)} rows to {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
