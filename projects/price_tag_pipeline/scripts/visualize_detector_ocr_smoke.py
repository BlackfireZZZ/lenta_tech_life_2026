#!/usr/bin/env python3
"""Smoke visualizer for raw zero-shot detections plus OCR.

This is intentionally separate from the final aggregation visualizer: it helps
debug whether detector candidates and OCR crops are alive before the full
tracking/dedup pipeline is tuned.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

THIS = Path(__file__).resolve().parent
SRC = THIS.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from price_tag_pipeline.ocr import PaddleOCREngine  # noqa: E402


DEFAULT_LABELS = (
    "price tag",
    "shelf price label",
    "barcode price label",
    "retail shelf label",
    "yellow price tag",
    "product label",
    "paper label",
    "text sign",
    "qr code",
)


@dataclass
class DrawDetection:
    bbox: tuple[int, int, int, int]
    label: str
    confidence: float
    ocr_text: str = ""
    ocr_confidence: float = 0.0


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _clip_box(box: tuple[int, int, int, int], w: int, h: int, pad: float = 0.08) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    px = int(round(bw * pad))
    py = int(round(bh * pad))
    return max(0, x1 - px), max(0, y1 - py), min(w - 1, x2 + px), min(h - 1, y2 + py)


def _short_text(text: str, limit: int = 80) -> str:
    text = " ".join((text or "").split())
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text


def _color(idx: int) -> tuple[int, int, int]:
    palette = (
        (44, 160, 255),
        (83, 220, 120),
        (255, 190, 70),
        (255, 100, 160),
        (170, 130, 255),
    )
    return palette[idx % len(palette)]


def _draw(frame: np.ndarray, detections: list[DrawDetection], frame_idx: int) -> np.ndarray:
    out = frame.copy()
    pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil, "RGBA")
    font = _load_font(24)
    small = _load_font(20)
    draw.rectangle((16, 16, 820, 92), fill=(0, 0, 0, 170))
    draw.text((28, 24), f"raw YOLO-World detections + PaddleOCR smoke | frame {frame_idx}", font=font, fill=(255, 255, 255, 255))
    draw.text((28, 58), "Boxes are detector candidates; OCR text is shown when crop OCR produced text.", font=small, fill=(210, 235, 255, 255))

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det.bbox
        c = _color(i)
        draw.rectangle((x1, y1, x2, y2), outline=(*c, 255), width=4)
        lines = [f"{det.label} {det.confidence:.2f}"]
        if det.ocr_text:
            lines.append(f"OCR {det.ocr_confidence:.2f}: {_short_text(det.ocr_text)}")
        text_w = max(draw.textbbox((0, 0), line, font=small)[2] for line in lines) + 14
        text_h = 26 * len(lines) + 10
        tx = max(0, min(x1, frame.shape[1] - text_w - 1))
        ty = max(96, y1 - text_h - 4)
        draw.rectangle((tx, ty, tx + text_w, ty + text_h), fill=(*c, 210))
        for j, line in enumerate(lines):
            draw.text((tx + 7, ty + 5 + j * 26), line, font=small, fill=(0, 0, 0, 255))

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _detections_from_result(result: Any, labels: tuple[str, ...], max_boxes: int, max_area_ratio: float) -> list[DrawDetection]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    h, w = result.orig_shape
    frame_area = float(w * h)
    dets: list[DrawDetection] = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = (int(round(float(v))) for v in boxes.xyxy[i].tolist())
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area <= 500 or area / frame_area > max_area_ratio:
            continue
        cls_id = int(boxes.cls[i].item()) if boxes.cls is not None else 0
        label = labels[cls_id] if 0 <= cls_id < len(labels) else str(cls_id)
        dets.append(DrawDetection((x1, y1, x2, y2), label, float(boxes.conf[i].item())))
    dets.sort(key=lambda d: d.confidence, reverse=True)
    return dets[:max_boxes]


def _ocr_best_orientation(ocr: PaddleOCREngine, crop: np.ndarray) -> tuple[str, float]:
    variants = (
        crop,
        cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE),
        cv2.rotate(crop, cv2.ROTATE_180),
    )
    best_text = ""
    best_conf = 0.0
    best_score = -1.0
    for img in variants:
        res = ocr.recognize(img)
        text = " ".join((res.text or "").split())
        score = len(text) * max(0.05, float(res.confidence))
        if score > best_score:
            best_text = text
            best_conf = float(res.confidence)
            best_score = score
    return best_text, best_conf


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="data/checkpoints/detector/yolov8s-worldv2.pt")
    p.add_argument("--conf", type=float, default=0.02)
    p.add_argument("--imgsz", type=int, default=1920)
    p.add_argument("--detect-stride", type=int, default=5)
    p.add_argument("--ocr-stride", type=int, default=25)
    p.add_argument("--max-boxes", type=int, default=18)
    p.add_argument("--max-ocr-per-frame", type=int, default=4)
    p.add_argument("--max-area-ratio", type=float, default=0.08)
    args = p.parse_args()

    from ultralytics import YOLO

    labels = DEFAULT_LABELS
    model = YOLO(args.model)
    model.set_classes(list(labels))
    ocr = PaddleOCREngine(lang="ru")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise SystemExit(f"Cannot open writer: {out_path}")

    last_dets: list[DrawDetection] = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % max(1, args.detect_stride) == 0:
            result = model.predict(frame, conf=args.conf, iou=0.5, imgsz=args.imgsz, verbose=False)[0]
            last_dets = _detections_from_result(result, labels, args.max_boxes, args.max_area_ratio)
            if frame_idx % max(1, args.ocr_stride) == 0:
                for det in last_dets[: args.max_ocr_per_frame]:
                    x1, y1, x2, y2 = _clip_box(det.bbox, w, h)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    try:
                        text, conf = _ocr_best_orientation(ocr, crop)
                    except Exception as exc:
                        det.ocr_text = f"OCR error: {exc}"
                        det.ocr_confidence = 0.0
                    else:
                        det.ocr_text = text
                        det.ocr_confidence = conf
        writer.write(_draw(frame, last_dets, frame_idx))
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Wrote {out_path} ({frame_idx} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
