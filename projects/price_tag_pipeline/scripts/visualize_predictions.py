#!/usr/bin/env python3
"""Render predicted bboxes + field values onto video frames; emit an annotated MP4.

Useful for jury demos and for debugging which observations made it into the
final report.

Usage:
    python projects/price_tag_pipeline/scripts/visualize_predictions.py \\
        --video  path/to/video.mp4 \\
        --pred   outputs/video01.jsonl \\
        --out    outputs/video01_annotated.mp4

The annotator picks one consistent random color per track_id; each tag's
fields appear as a small label box near its bbox.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _color_for_track(track_id: int) -> tuple[int, int, int]:
    rng = random.Random(track_id * 9973 + 17)
    return rng.randint(60, 255), rng.randint(60, 255), rng.randint(60, 255)


def _format_label(tag: dict[str, Any]) -> list[str]:
    lines = []
    if tag.get("regular_price") is not None:
        lines.append(f"reg: {tag['regular_price']:.2f} {tag.get('currency', 'RUB')}")
    if tag.get("loyalty_price") is not None:
        lines.append(f"loy: {tag['loyalty_price']:.2f}")
    name = tag.get("product_name")
    if name:
        if len(name) > 32:
            name = name[:31] + "…"
        lines.append(name)
    if tag.get("weight_value") is not None and tag.get("weight_unit"):
        lines.append(f"{tag['weight_value']:g} {tag['weight_unit']}")
    if tag.get("promo_flag"):
        lines.append("PROMO")
    if tag.get("qr_code_barcode"):
        lines.append(f"QR: {tag['qr_code_barcode']}")
    return lines


def _draw_tag(frame: np.ndarray, tag: dict[str, Any]) -> None:
    bbox = tag.get("bbox")
    if not bbox or len(bbox) != 4:
        return
    x1, y1, x2, y2 = (int(v) for v in bbox)
    color = _color_for_track(int(tag.get("track_id", 0)))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label_lines = _format_label(tag)
    if not label_lines:
        return
    # Label background
    text_h = 18
    pad = 4
    label_w = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
                  for line in label_lines) + 2 * pad
    label_h = text_h * len(label_lines) + 2 * pad
    ly1 = max(0, y1 - label_h - 2)
    cv2.rectangle(frame, (x1, ly1), (x1 + label_w, ly1 + label_h), color, -1)
    for i, line in enumerate(label_lines):
        cv2.putText(frame, line, (x1 + pad, ly1 + pad + (i + 1) * text_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _format_audit(row: dict[str, Any]) -> str:
    parsed = row.get("parsed") or {}
    bits = [str(row.get("backend", "ocr"))]
    if parsed.get("qr_code_barcode") or (parsed.get("extra_fields") or {}).get("qr_code_barcode"):
        bits.append("QR")
    for key in ("regular_price", "loyalty_price", "product_name"):
        value = parsed.get(key)
        if value not in (None, ""):
            bits.append(f"{key}={value}")
    extras = parsed.get("extra_fields") or {}
    for key in ("barcode", "price1_qr", "price2_qr", "action_price_qr"):
        if extras.get(key) not in (None, ""):
            bits.append(f"{key}={extras[key]}")
    text = " | ".join(bits)
    return text[:120]


def _draw_audit_panel(frame: np.ndarray, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    lines = ["OCR/QR read:"] + [_format_audit(r) for r in rows[:5]]
    pad = 8
    line_h = 20
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    width = min(
        frame.shape[1] - 20,
        max(cv2.getTextSize(line, font, scale, 1)[0][0] for line in lines) + 2 * pad,
    )
    height = line_h * len(lines) + 2 * pad
    x1, y1 = 10, 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x1 + width, y1 + height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, dst=frame)
    for i, line in enumerate(lines):
        color = (80, 240, 255) if i == 0 else (255, 255, 255)
        cv2.putText(frame, line, (x1 + pad, y1 + pad + (i + 1) * line_h - 5),
                    font, scale, color, 1, cv2.LINE_AA)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--pred", required=True, help="JSONL of FinalTag predictions")
    p.add_argument("--out", required=True, help="Output annotated MP4")
    p.add_argument("--audit", default=None, help="Optional OCR/QR audit JSONL from config runtime.audit_path")
    p.add_argument("--fourcc", default="mp4v")
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--linger-frames", type=int, default=45,
                   help="Keep each final tag visible for N frames after source frames")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO)
    tags = _load_jsonl(Path(args.pred))
    # Index tags by source frame for O(1) lookup.
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for t in tags:
        for f in t.get("source_frames", []):
            start = int(f)
            for ff in range(start, start + max(1, args.linger_frames + 1)):
                by_frame.setdefault(ff, []).append(t)
    logging.info("Loaded %d tags spanning %d unique frames", len(tags), len(by_frame))

    audit_by_frame: dict[int, list[dict[str, Any]]] = {}
    if args.audit:
        audit_rows = _load_jsonl(Path(args.audit))
        for row in audit_rows:
            if row.get("frame_idx") is not None:
                audit_by_frame.setdefault(int(row["frame_idx"]), []).append(row)
        logging.info("Loaded %d OCR/QR audit events", len(audit_rows))

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*args.fourcc)
    writer = cv2.VideoWriter(args.out, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise SystemExit(f"Cannot open writer for {args.out} (try a different --fourcc)")

    frame_idx = 0
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % args.frame_stride == 0:
            for t in by_frame.get(frame_idx, []):
                _draw_tag(frame, t)
            _draw_audit_panel(frame, audit_by_frame.get(frame_idx, []))
            writer.write(frame)
            written += 1
        frame_idx += 1
    cap.release()
    writer.release()
    logging.info("Wrote %s (%d frames)", args.out, written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
