#!/usr/bin/env python3
"""CVAT → ours: convert a CVAT export into the canonical raw layout.

Auto-detects the export kind from the XML and the labelling style:

* **video, fields filled** (``--mode tags``) → 29-column Lenta CSV (one row
  per track = one tag, ``frame_timestamp`` in **ms**, ``track_id`` column =
  free cross-track-dedup ground truth) into
  ``<out-raw>/annotations/csv/<id>.csv`` (+ stages the video). Feeds the
  full OCR/E2E pipeline via ``prepare_data.py`` unchanged.

* **video, boxes only** (``--mode detector``) → **dense** YOLO labels:
  every track (the noisy seed + every tag you add) is **interpolated
  between its keyframes**, so 2 keyframes per tag yield a labelled box on
  every frame in between. Written as the YOLO raw layout under
  ``<out-raw>`` (default ``data/raw_det``) + only the labelled frames
  extracted — no OCR fields needed. This is the path for "just draw boxes,
  improve the detector".

* **images** (photos) → YOLO raw layout under ``data/raw_photos``.

``--mode auto`` (default): if no track/box carries any substantive field,
use ``detector``; otherwise ``tags``.

    # boxes-only video → dense detector data (auto-detected)
    python projects/price_tag_pipeline/scripts/cvat_import.py \\
        --xml export/25_2-10.xml \\
        --video real_data/dataset/25_2-10/25_2-10.mp4

    # store photos
    python projects/price_tag_pipeline/scripts/cvat_import.py \\
        --xml export/shots.xml --images-dir my_photos --set store_run_1
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
SRC = THIS.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None  # type: ignore[assignment]

from price_tag_pipeline.cv_io import imread, imwrite  # noqa: E402
from price_tag_pipeline.data.cvat import (  # noqa: E402
    CvatDoc,
    cvat_images_to_yolo,
    cvat_video_to_lenta_rows,
    cvat_video_to_yolo,
    has_substantive_annotations,
    lenta_rows_to_csv_text,
    parse_cvat_xml,
)
from price_tag_pipeline.data.loaders import _read_frame_near  # noqa: E402

LOGGER = logging.getLogger("cvat_import")


def _open_video(video: Path):
    if cv2 is None:
        raise ModuleNotFoundError("Video import needs OpenCV; install opencv-python.")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise OSError(f"Could not open {video}")
    return cap


def _import_video_tags(doc: CvatDoc, video: Path, out_raw: Path, video_id: str) -> int:
    cap = _open_video(video)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    rows = cvat_video_to_lenta_rows(doc, fps=fps, filename=video_id)
    if not rows:
        LOGGER.error("No usable tracks for tags mode")
        return 1
    csv_dir = out_raw / "annotations" / "csv"
    videos_dir = out_raw / "videos"
    csv_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    (csv_dir / f"{video_id}.csv").write_text(lenta_rows_to_csv_text(rows), encoding="utf-8")
    dst = videos_dir / video.name
    if not dst.exists() or dst.stat().st_size != video.stat().st_size:
        shutil.copy2(video, dst)
    LOGGER.info("tags: wrote %s (%d tags, fps=%.3f) + staged %s",
                csv_dir / f"{video_id}.csv", len(rows), fps, dst)
    LOGGER.info("Next: prepare_data.py --raw %s --processed data/processed", out_raw)
    return 0


def _import_video_detector(doc: CvatDoc, video: Path, out_raw: Path, video_id: str) -> int:
    cap = _open_video(video)
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fcount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = doc.width or vw
    height = doc.height or vh
    labels = cvat_video_to_yolo(doc, width=width, height=height)
    if not labels:
        cap.release()
        LOGGER.error("No visible boxes in any track — nothing to write")
        return 1

    frames_dir = out_raw / "frames" / video_id
    labels_dir = out_raw / "annotations" / "labels" / video_id
    frames_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    (out_raw / "annotations" / "classes.txt").write_text("price_tag\n", encoding="utf-8")

    written = 0
    for f, lines in sorted(labels.items()):
        dst_img = frames_dir / f"{f:06d}.jpg"
        if not dst_img.exists():
            frame = _read_frame_near(cap, f, frame_count=fcount)
            if frame is None:
                LOGGER.warning("Could not extract frame %d — skipped", f)
                continue
            imwrite(dst_img, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (labels_dir / f"{f:06d}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        written += 1
    cap.release()

    n_boxes = sum(len(v) for v in labels.values())
    LOGGER.info("detector: %s -> %d labelled frames, %d boxes (%dx%d) in %s",
                video_id, written, n_boxes, width, height, out_raw.resolve())
    LOGGER.info("Next: prepare_data.py --raw %s --processed data/processed_det", out_raw)
    return 0 if written else 1


def _import_images(doc: CvatDoc, images_dir: Path, set_name: str, out_raw: Path) -> int:
    frames_dir = out_raw / "frames" / set_name
    labels_dir = out_raw / "annotations" / "labels" / set_name
    frames_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    (out_raw / "annotations" / "classes.txt").write_text("price_tag\n", encoding="utf-8")

    per_image = cvat_images_to_yolo(doc)
    mapping = ["index,source_name"]
    written = 0
    for idx, im in enumerate(doc.images):
        w, h, lines = per_image.get(im.name, (0, 0, []))
        src = images_dir / Path(im.name).name
        if not src.exists():
            LOGGER.warning("Image %s not found under %s — skipped", im.name, images_dir)
            continue
        stem = f"{idx:06d}"
        dst_img = frames_dir / f"{stem}.jpg"
        if src.suffix.lower() in (".jpg", ".jpeg"):
            shutil.copy2(src, dst_img)
        else:
            img = imread(src)
            if img is None:
                LOGGER.warning("Cannot read %s — skipped", src)
                continue
            imwrite(dst_img, img)
        (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        mapping.append(f"{stem},{im.name}")
        written += 1

    (out_raw / "frames" / f"{set_name}__name_map.csv").write_text(
        "\n".join(mapping) + "\n", encoding="utf-8"
    )
    LOGGER.info("images: %d photos+labels for set '%s' -> %s",
                written, set_name, out_raw.resolve())
    LOGGER.info("Next: prepare_data.py --raw %s --processed data/processed_photos", out_raw)
    return 0 if written else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--xml", type=Path, required=True, help="CVAT export XML")
    p.add_argument("--video", type=Path, help="Source video (video task)")
    p.add_argument("--images-dir", type=Path, help="Folder of source photos (image task)")
    p.add_argument("--set", dest="set_name", help="Photo set id (image task)")
    p.add_argument("--mode", choices=["auto", "tags", "detector"], default="auto",
                   help="auto: detector if no fields filled, else tags (video only)")
    p.add_argument("--out-raw", type=Path, help="Target raw root (default depends on mode/kind)")
    p.add_argument("--id", dest="video_id", help="video_id (defaults to video stem)")
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args()
    logging.basicConfig(level=getattr(logging, a.log_level), format="%(levelname)s: %(message)s")

    if not a.xml.exists():
        LOGGER.error("XML not found: %s", a.xml)
        return 2

    doc = parse_cvat_xml(a.xml.read_text(encoding="utf-8"))

    if a.video:
        if doc.mode != "video":
            LOGGER.error("XML has no <track> — this is an image export; use --images-dir")
            return 1
        mode = a.mode
        if mode == "auto":
            mode = "tags" if has_substantive_annotations(doc) else "detector"
            LOGGER.info("auto mode -> %s (substantive fields %s)",
                        mode, "present" if mode == "tags" else "absent")
        vid = a.video_id or a.video.stem
        if mode == "tags":
            out_raw = a.out_raw or Path("data/raw")
            return _import_video_tags(doc, a.video, out_raw, vid)
        out_raw = a.out_raw or Path("data/raw_det")
        return _import_video_detector(doc, a.video, out_raw, vid)

    if a.images_dir:
        if doc.mode != "images":
            LOGGER.error("XML has no <image> — this is a video export; use --video")
            return 1
        if not a.set_name:
            LOGGER.error("--set NAME is required with --images-dir")
            return 2
        out_raw = a.out_raw or Path("data/raw_photos")
        return _import_images(doc, a.images_dir, a.set_name, out_raw)

    LOGGER.error("Pass --video PATH (video task) or --images-dir DIR --set NAME (photos)")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
