#!/usr/bin/env python3
"""Export per-video JSONL predictions to the hackathon CSV schema.

Input:
  Directory with *.jsonl files produced by run_inference / run_batch_inference.

Output:
  One CSV where each row is one predicted price tag.

Notes:
  - Fields unsupported by the current baseline are emitted as empty strings.
  - `frame_timestamp` is emitted in milliseconds.
  - `filename` defaults to the JSONL stem because the released Lenta CSVs use
    stems such as `25_2-10`, not `25_2-10.mp4`.
  - `price_discount` is derived as max(price_default - price_card, 0) when both
    prices are present; otherwise empty.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

CSV_COLUMNS = [
    "filename",
    "product_name",
    "price_default",
    "price_card",
    "price_discount",
    "barcode",
    "discount_amount",
    "id_sku",
    "print_datetime",
    "code",
    "additional_info",
    "color",
    "special_symbols",
    "frame_timestamp",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "qr_code_barcode",
    "price1_qr",
    "price2_qr",
    "price3_qr",
    "price4_qr",
    "wholesale_level_1_count",
    "wholesale_level_1_price",
    "wholesale_level_2_count",
    "wholesale_level_2_price",
    "action_price_qr",
    "action_code_qr",
]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _fmt_price(v: Any) -> str:
    if v is None:
        return ""
    try:
        return f"{float(v):.2f}"
    except (TypeError, ValueError):
        return ""


def _fmt_cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def _as_int_str(v: Any) -> str:
    if v is None:
        return ""
    try:
        return str(int(round(float(v))))
    except (TypeError, ValueError):
        return ""


def _derive_discount(regular: float | None, card: float | None) -> str:
    if regular is None or card is None:
        return ""
    diff = float(regular) - float(card)
    if diff <= 0:
        return "0.00"
    return f"{diff:.2f}"


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _row_from_tag(tag: dict[str, Any], filename: str) -> dict[str, str]:
    bbox = tag.get("bbox") or [None, None, None, None]
    if len(bbox) != 4:
        bbox = [None, None, None, None]

    regular = _to_float(tag.get("regular_price"))
    card = _to_float(tag.get("loyalty_price"))
    ts_ms = None
    if tag.get("timestamp_s") is not None:
        try:
            ts_ms = int(round(float(tag["timestamp_s"]) * 1000.0))
        except (TypeError, ValueError):
            ts_ms = None

    price_discount = tag.get("price_discount")
    if price_discount in (None, ""):
        price_discount = _derive_discount(regular, card)

    row = {
        "filename": filename,
        "product_name": str(tag.get("product_name") or ""),
        "price_default": _fmt_price(regular),
        "price_card": _fmt_price(card),
        "price_discount": _fmt_cell(price_discount),
        "barcode": _fmt_cell(tag.get("barcode")),
        "discount_amount": _fmt_cell(tag.get("discount_amount")),
        "id_sku": _fmt_cell(tag.get("id_sku")),
        "print_datetime": _fmt_cell(tag.get("print_datetime")),
        "code": _fmt_cell(tag.get("code")),
        "additional_info": _fmt_cell(tag.get("additional_info")),
        "color": _fmt_cell(tag.get("color")),
        "special_symbols": _fmt_cell(tag.get("special_symbols")),
        "frame_timestamp": _as_int_str(ts_ms),
        "x_min": _as_int_str(bbox[0]),
        "y_min": _as_int_str(bbox[1]),
        "x_max": _as_int_str(bbox[2]),
        "y_max": _as_int_str(bbox[3]),
        "qr_code_barcode": _fmt_cell(tag.get("qr_code_barcode")),
        "price1_qr": _fmt_cell(tag.get("price1_qr")),
        "price2_qr": _fmt_cell(tag.get("price2_qr")),
        "price3_qr": _fmt_cell(tag.get("price3_qr")),
        "price4_qr": _fmt_cell(tag.get("price4_qr")),
        "wholesale_level_1_count": _fmt_cell(tag.get("wholesale_level_1_count")),
        "wholesale_level_1_price": _fmt_cell(tag.get("wholesale_level_1_price")),
        "wholesale_level_2_count": _fmt_cell(tag.get("wholesale_level_2_count")),
        "wholesale_level_2_price": _fmt_cell(tag.get("wholesale_level_2_price")),
        "action_price_qr": _fmt_cell(tag.get("action_price_qr")),
        "action_code_qr": _fmt_cell(tag.get("action_code_qr")),
    }
    return row


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
        tags = _read_jsonl(jf)
        for tag in tags:
            rows_out.append(_row_from_tag(tag, filename=video_filename))

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Wrote {len(rows_out)} rows to {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
