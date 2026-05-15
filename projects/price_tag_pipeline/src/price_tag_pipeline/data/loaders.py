"""Format adapters: YOLO ↔ unified internal representation, with converters.

Unified internal representation is YOLO normalized (cx, cy, w, h) per-image txt
under data/processed/labels/{video_id}/{frame_idx:06d}.txt and JPEG frames under
data/processed/frames/{video_id}/{frame_idx:06d}.jpg.
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2

LOGGER = logging.getLogger(__name__)


@dataclass
class FrameRecord:
    video_id: str
    frame_idx: int
    image_path: Path
    label_path: Path


def detect_format(raw_dir: Path) -> str:
    """Return one of: 'yolo', 'coco', 'lenta_csv', 'frames_only', 'empty'."""
    coco = list((raw_dir / "annotations").glob("*.json"))
    if coco:
        return "coco"
    csv_dir = raw_dir / "annotations" / "csv"
    csv_files = list(csv_dir.glob("*.csv")) if csv_dir.exists() else list((raw_dir / "annotations").glob("*.csv"))
    if csv_files:
        return "lenta_csv"
    if (raw_dir / "annotations" / "labels").exists():
        return "yolo"
    if (raw_dir / "frames").exists():
        return "frames_only"
    return "empty"


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_frames(video_path: Path, out_dir: Path, jpeg_quality: int = 95) -> int:
    """Extract every frame from `video_path` into `out_dir / {frame_idx:06d}.jpg`.

    Returns the number of frames written. Skips work if the directory already
    has the expected number of files (idempotent).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    existing = len(list(out_dir.glob("*.jpg")))
    if total > 0 and existing == total:
        cap.release()
        LOGGER.info("Skipping %s: %d frames already extracted", video_path.name, existing)
        return existing

    idx = 0
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(str(out_dir / f"{idx:06d}.jpg"), frame, encode_params)
        idx += 1
    cap.release()
    return idx


def load_classes(raw_dir: Path) -> list[str]:
    classes_file = raw_dir / "annotations" / "classes.txt"
    if classes_file.exists():
        return [
            line.strip()
            for line in classes_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    # Default: single class.
    return ["price_tag"]


# ---------------------------------------------------------------------------
# YOLO -> processed
# ---------------------------------------------------------------------------

def ingest_yolo(
    raw_dir: Path,
    processed_dir: Path,
    extract_videos: bool = True,
) -> tuple[list[FrameRecord], list[str]]:
    """Walk a YOLO-formatted raw_dir and stage frames + labels into processed_dir.

    Expected raw layout:
        raw_dir/videos/{video_id}.(mp4|avi|mov|mkv)
        raw_dir/frames/{video_id}/{frame_idx:06d}.jpg    (alternative to videos)
        raw_dir/annotations/labels/{video_id}/{frame_idx:06d}.txt
        raw_dir/annotations/classes.txt (optional)

    Returns (FrameRecord list, classes).
    """
    classes = load_classes(raw_dir)
    labels_root = raw_dir / "annotations" / "labels"
    videos_dir = raw_dir / "videos"
    raw_frames_dir = raw_dir / "frames"
    out_frames_dir = processed_dir / "frames"
    out_labels_dir = processed_dir / "labels"
    out_frames_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    # Discover video IDs from label folders (authoritative — only labeled frames matter).
    video_ids = sorted(p.name for p in labels_root.iterdir() if p.is_dir()) if labels_root.exists() else []
    if not video_ids:
        LOGGER.warning("No video subfolders found under %s", labels_root)
        return [], classes

    records: list[FrameRecord] = []
    for vid in video_ids:
        # 1. Source frames: either pre-extracted or from a video file.
        src_frames = raw_frames_dir / vid
        if src_frames.exists():
            staged_frames = src_frames
        elif extract_videos and videos_dir.exists():
            video_file = _find_video_file(videos_dir, vid)
            if video_file is None:
                LOGGER.warning("No video file found for %s under %s", vid, videos_dir)
                continue
            staged_frames = out_frames_dir / vid
            extract_frames(video_file, staged_frames)
        else:
            LOGGER.warning("No frames source for %s — skipping", vid)
            continue

        # 2. Mirror labels into processed/labels and pair with frames.
        label_dir = labels_root / vid
        out_label_dir = out_labels_dir / vid
        out_label_dir.mkdir(parents=True, exist_ok=True)

        out_frame_dir = out_frames_dir / vid
        out_frame_dir.mkdir(parents=True, exist_ok=True)

        for label_file in sorted(label_dir.glob("*.txt")):
            stem = label_file.stem  # "000123"
            try:
                frame_idx = int(stem)
            except ValueError:
                LOGGER.warning("Skipping non-numeric label name: %s", label_file)
                continue
            src_img = staged_frames / f"{stem}.jpg"
            if not src_img.exists():
                # Some datasets ship .png — try.
                alt = list(staged_frames.glob(f"{stem}.*"))
                if not alt:
                    LOGGER.warning("Label %s has no matching image — skipped", label_file)
                    continue
                src_img = alt[0]
            dst_img = out_frame_dir / f"{stem}.jpg"
            dst_label = out_label_dir / f"{stem}.txt"
            if not dst_img.exists():
                if src_img.suffix.lower() == ".jpg":
                    if src_img != dst_img:
                        shutil.copy2(src_img, dst_img)
                else:
                    img = cv2.imread(str(src_img))
                    if img is None:
                        LOGGER.warning("Could not read %s — skipped", src_img)
                        continue
                    cv2.imwrite(str(dst_img), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not dst_label.exists():
                shutil.copy2(label_file, dst_label)
            records.append(FrameRecord(video_id=vid, frame_idx=frame_idx, image_path=dst_img, label_path=dst_label))

    LOGGER.info("Ingested %d frame records across %d videos", len(records), len(video_ids))
    return records, classes


def _find_video_file(videos_dir: Path, video_id: str) -> Optional[Path]:
    for ext in (".mp4", ".avi", ".mov", ".mkv"):
        p = videos_dir / f"{video_id}{ext}"
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Lenta CSV -> processed YOLO + E2E ground truth
# ---------------------------------------------------------------------------

def ingest_lenta_csv(
    raw_dir: Path,
    processed_dir: Path,
    extract_videos: bool = True,
) -> tuple[list[FrameRecord], list[str]]:
    """Convert the hackathon CSV annotation format into processed YOLO files.

    Expected raw layout:
        raw_dir/videos/{video_id}.mp4
        raw_dir/annotations/csv/{video_id}.csv

    The CSV stores one price-tag row per object, with bbox columns
    `x_min,y_min,x_max,y_max` and the source frame in `frame_timestamp`.
    We extract only the annotated frames, aggregate all boxes that share a
    frame, and write one YOLO label file per frame.
    """
    if not extract_videos:
        raise ValueError("lenta_csv ingestion needs video frame extraction; remove --no-extract")

    classes = ["price_tag"]
    csv_root = raw_dir / "annotations" / "csv"
    csv_files = sorted(csv_root.glob("*.csv")) if csv_root.exists() else sorted((raw_dir / "annotations").glob("*.csv"))
    csv_files = [p for p in csv_files if p.name.lower() != "sample.csv"]

    videos_dir = raw_dir / "videos"
    out_frames_dir = processed_dir / "frames"
    out_labels_dir = processed_dir / "labels"
    gt_dir = processed_dir / "gt_e2e"
    out_frames_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    records: list[FrameRecord] = []
    for csv_path in csv_files:
        video_id = csv_path.stem
        rows = _read_lenta_csv(csv_path)
        if not rows:
            LOGGER.warning("Skipping empty CSV: %s", csv_path)
            continue

        video_path = _find_lenta_video_file(videos_dir, video_id, rows)
        if video_path is None:
            LOGGER.warning("No video found for %s under %s", csv_path.name, videos_dir)
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            LOGGER.warning("Could not open %s", video_path)
            continue
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            cap.release()
            LOGGER.warning("Could not read video dimensions for %s", video_path)
            continue

        labels_by_frame: dict[int, list[str]] = {}
        gt_rows: list[dict[str, object]] = []
        for row_no, row in enumerate(rows, start=2):
            frame_idx = _parse_int(row.get("frame_timestamp"))
            box = _parse_lenta_bbox(row, width, height)
            if frame_idx is None or box is None:
                LOGGER.warning("Skipping bad row %s:%d", csv_path, row_no)
                continue

            x_min, y_min, x_max, y_max = box
            cx = ((x_min + x_max) / 2) / width
            cy = ((y_min + y_max) / 2) / height
            bw = (x_max - x_min) / width
            bh = (y_max - y_min) / height
            labels_by_frame.setdefault(frame_idx, []).append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            gt = dict(row)
            gt["video_id"] = video_id
            gt["frame_idx"] = frame_idx
            gt["bbox"] = [round(x_min, 2), round(y_min, 2), round(x_max, 2), round(y_max, 2)]
            gt_rows.append(gt)

        out_frame_dir = out_frames_dir / video_id
        out_label_dir = out_labels_dir / video_id
        out_frame_dir.mkdir(parents=True, exist_ok=True)
        out_label_dir.mkdir(parents=True, exist_ok=True)

        for frame_idx, lines in sorted(labels_by_frame.items()):
            dst_img = out_frame_dir / f"{frame_idx:06d}.jpg"
            dst_label = out_label_dir / f"{frame_idx:06d}.txt"
            if not dst_img.exists():
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok or frame is None:
                    LOGGER.warning("Could not extract frame %d from %s", frame_idx, video_path)
                    continue
                cv2.imwrite(str(dst_img), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            dst_label.write_text("\n".join(lines) + "\n", encoding="utf-8")
            records.append(
                FrameRecord(video_id=video_id, frame_idx=frame_idx, image_path=dst_img, label_path=dst_label)
            )

        cap.release()
        gt_path = gt_dir / f"{video_id}.jsonl"
        with gt_path.open("w", encoding="utf-8") as f:
            for row in gt_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        LOGGER.info(
            "Converted %s: %d rows, %d labeled frames, gt=%s",
            csv_path.name,
            len(gt_rows),
            len(labels_by_frame),
            gt_path,
        )

    LOGGER.info("Converted %d labeled frames from %d Lenta CSV files", len(records), len(csv_files))
    return records, classes


def _read_lenta_csv(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _find_lenta_video_file(videos_dir: Path, video_id: str, rows: list[dict[str, str]]) -> Optional[Path]:
    direct = _find_video_file(videos_dir, video_id)
    if direct is not None:
        return direct

    candidates = [video_id]
    for row in rows[:10]:
        filename = (row.get("filename") or "").strip()
        if not filename:
            continue
        candidates.append(Path(filename).stem)
        candidates.append(Path(filename).parent.name)

    for candidate in candidates:
        if not candidate or candidate == ".":
            continue
        found = _find_video_file(videos_dir, candidate)
        if found is not None:
            return found
    return None


def _parse_decimal(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().replace("\xa0", "").replace(" ", "").replace(",", ".")
    if not text or text.lower() == "нет":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_int(value: object) -> Optional[int]:
    parsed = _parse_decimal(value)
    if parsed is None:
        return None
    return int(round(parsed))


def _parse_lenta_bbox(row: dict[str, str], width: int, height: int) -> Optional[tuple[float, float, float, float]]:
    values = [_parse_decimal(row.get(k)) for k in ("x_min", "y_min", "x_max", "y_max")]
    if any(v is None for v in values):
        return None
    x_min, y_min, x_max, y_max = (float(v) for v in values if v is not None)
    x_min = max(0.0, min(float(width), x_min))
    x_max = max(0.0, min(float(width), x_max))
    y_min = max(0.0, min(float(height), y_min))
    y_max = max(0.0, min(float(height), y_max))
    if x_max <= x_min or y_max <= y_min:
        return None
    return x_min, y_min, x_max, y_max


# ---------------------------------------------------------------------------
# COCO -> YOLO converter
# ---------------------------------------------------------------------------

def coco_to_yolo(
    raw_dir: Path,
    processed_dir: Path,
    coco_json: Optional[Path] = None,
) -> tuple[list[FrameRecord], list[str]]:
    """Convert a COCO-format annotations file into YOLO-format files under processed/.

    Expects image filenames in the COCO json to look like `{video_id}_{frame_idx:06d}.jpg`
    or `{video_id}/{frame_idx:06d}.jpg`. Frames must already exist under
    raw_dir/frames or raw_dir/videos (we will extract from videos if needed).
    """
    if coco_json is None:
        candidates = list((raw_dir / "annotations").glob("*.json"))
        if not candidates:
            raise FileNotFoundError("No COCO JSON found under raw/annotations/")
        coco_json = candidates[0]

    with open(coco_json, encoding="utf-8") as f:
        coco = json.load(f)

    categories = sorted(coco.get("categories", []), key=lambda c: c["id"])
    classes = [c["name"] for c in categories]
    cat_id_to_yolo_idx = {c["id"]: i for i, c in enumerate(categories)}

    images_by_id = {img["id"]: img for img in coco["images"]}
    anns_by_img: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    out_frames_dir = processed_dir / "frames"
    out_labels_dir = processed_dir / "labels"

    records: list[FrameRecord] = []
    for img_id, img in images_by_id.items():
        filename = img["file_name"]
        # Try patterns to recover video_id and frame_idx
        video_id, frame_idx = _parse_coco_filename(filename)
        if video_id is None:
            LOGGER.warning("Cannot parse video_id/frame_idx from %s — skipped", filename)
            continue
        out_frame_dir = out_frames_dir / video_id
        out_label_dir = out_labels_dir / video_id
        out_frame_dir.mkdir(parents=True, exist_ok=True)
        out_label_dir.mkdir(parents=True, exist_ok=True)

        dst_img = out_frame_dir / f"{frame_idx:06d}.jpg"
        dst_label = out_label_dir / f"{frame_idx:06d}.txt"

        if not dst_img.exists():
            src = _locate_coco_source_image(raw_dir, filename, video_id, frame_idx)
            if src is None:
                LOGGER.warning("Source image not found for %s", filename)
                continue
            shutil.copy2(src, dst_img)

        w, h = img["width"], img["height"]
        lines = []
        for ann in anns_by_img.get(img_id, []):
            cat = cat_id_to_yolo_idx.get(ann["category_id"])
            if cat is None:
                continue
            x, y, bw, bh = ann["bbox"]  # COCO: top-left + wh
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h
            lines.append(f"{cat} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        dst_label.write_text("\n".join(lines), encoding="utf-8")

        records.append(FrameRecord(video_id=video_id, frame_idx=frame_idx, image_path=dst_img, label_path=dst_label))

    LOGGER.info("Converted %d COCO images to YOLO format", len(records))
    return records, classes


def _parse_coco_filename(fname: str) -> tuple[Optional[str], Optional[int]]:
    stem = Path(fname).stem
    # 'store_07_aisle_03/000123'
    if "/" in fname:
        parts = fname.split("/")
        try:
            return parts[-2], int(Path(parts[-1]).stem)
        except (IndexError, ValueError):
            return None, None
    # 'store_07_aisle_03_000123'
    if "_" in stem:
        head, _, tail = stem.rpartition("_")
        try:
            return head, int(tail)
        except ValueError:
            return None, None
    return None, None


def _locate_coco_source_image(
    raw_dir: Path,
    fname: str,
    video_id: str,
    frame_idx: int,
) -> Optional[Path]:
    candidates = [
        raw_dir / "frames" / fname,
        raw_dir / "frames" / video_id / f"{frame_idx:06d}.jpg",
        raw_dir / "frames" / video_id / f"{frame_idx:06d}.png",
        raw_dir / "annotations" / "images" / fname,
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


# ---------------------------------------------------------------------------
# Dataset YAML emission (Ultralytics-compatible)
# ---------------------------------------------------------------------------

def write_dataset_yaml(
    processed_dir: Path,
    classes: list[str],
    fold: Optional[dict[str, list[str]]] = None,
    fold_id: int = 0,
) -> Path:
    """Emit dataset.yaml. `fold`, if provided, defines `train`/`val` video_id lists."""
    yaml_path = processed_dir / "dataset.yaml"
    lines = [
        f"path: {processed_dir.resolve().as_posix()}",
        "train: train_images.txt",
        "val: val_images.txt",
        f"nc: {len(classes)}",
        "names:",
    ]
    for i, name in enumerate(classes):
        lines.append(f"  {i}: {name}")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if fold is not None:
        _emit_split_list(processed_dir, "train_images.txt", fold["train"])
        _emit_split_list(processed_dir, "val_images.txt", fold["val"])
    return yaml_path


def _emit_split_list(processed_dir: Path, fname: str, video_ids: list[str]) -> None:
    frames_root = processed_dir / "frames"
    out = processed_dir / fname
    lines: list[str] = []
    for vid in video_ids:
        for img in sorted((frames_root / vid).glob("*.jpg")):
            lines.append(img.resolve().as_posix())
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
