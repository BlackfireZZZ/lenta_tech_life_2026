#!/usr/bin/env python3
"""CVAT → ours: convert a CVAT export into the canonical ``data/raw`` layout.

Auto-detects the export kind from the XML:

* **video** (``<track>`` present) → writes a 29-column Lenta CSV
  (one row per track = one physical tag, ``frame_timestamp`` in **ms**,
  plus a ``track_id`` column = free cross-track-dedup ground truth) into
  ``<out-raw>/annotations/csv/<id>.csv`` and copies the video into
  ``<out-raw>/videos/``. ``prepare_data.py`` then ingests it **unchanged**.

* **images** (``<image>`` present) → writes the YOLO raw layout
  (``annotations/labels/<set>/*.txt`` + ``frames/<set>/*.jpg`` +
  ``classes.txt``) into ``<out-raw>`` (default ``data/raw_photos`` so a
  photo set never shadows a video CSV under the same root).

Usage:

    # video task export
    python projects/price_tag_pipeline/scripts/cvat_import.py \\
        --xml export/25_2-10.xml --video real_data/dataset/25_2-10/25_2-10.mp4

    # store-photo task export
    python projects/price_tag_pipeline/scripts/cvat_import.py \\
        --xml export/store_photos.xml --images-dir my_photos --set store_run_1
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
    cvat_video_to_lenta_rows,
    lenta_rows_to_csv_text,
    parse_cvat_xml,
)

LOGGER = logging.getLogger("cvat_import")


def _video_fps(video: Path) -> float:
    if cv2 is None:
        raise ModuleNotFoundError("Video import needs OpenCV; install opencv-python.")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise OSError(f"Could not open {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    return fps


def _import_video(xml: Path, video: Path, out_raw: Path, video_id: str) -> int:
    doc = parse_cvat_xml(xml.read_text(encoding="utf-8"))
    if doc.mode != "video":
        LOGGER.error("XML is not a video export (no <track>). Use --images-dir.")
        return 1
    fps = _video_fps(video)
    rows = cvat_video_to_lenta_rows(doc, fps=fps, filename=video_id)
    if not rows:
        LOGGER.error("No usable tracks in %s", xml)
        return 1

    csv_dir = out_raw / "annotations" / "csv"
    videos_dir = out_raw / "videos"
    csv_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    csv_out = csv_dir / f"{video_id}.csv"
    csv_out.write_text(lenta_rows_to_csv_text(rows), encoding="utf-8")
    dst_video = videos_dir / video.name
    if not dst_video.exists() or dst_video.stat().st_size != video.stat().st_size:
        shutil.copy2(video, dst_video)

    LOGGER.info("Wrote %s (%d tags, fps=%.3f) + staged %s",
                csv_out, len(rows), fps, dst_video)
    LOGGER.info("Next: python projects/price_tag_pipeline/scripts/prepare_data.py "
                "--raw %s --processed data/processed", out_raw)
    return 0


def _import_images(xml: Path, images_dir: Path, set_name: str, out_raw: Path) -> int:
    doc = parse_cvat_xml(xml.read_text(encoding="utf-8"))
    if doc.mode != "images":
        LOGGER.error("XML is not an image export (no <image>). Use --video.")
        return 1

    frames_dir = out_raw / "frames" / set_name
    labels_dir = out_raw / "annotations" / "labels" / set_name
    frames_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    (out_raw / "annotations" / "classes.txt").write_text("price_tag\n", encoding="utf-8")

    mapping: list[str] = ["index,source_name"]
    written = 0
    for idx, im in enumerate(doc.images):
        src = images_dir / Path(im.name).name
        if not src.exists():
            LOGGER.warning("Image %s not found under %s — skipped", im.name, images_dir)
            continue
        w = im.width or doc.width
        h = im.height or doc.height
        if w <= 0 or h <= 0:
            img = imread(src)
            if img is None:
                LOGGER.warning("Cannot read %s — skipped", src)
                continue
            h, w = img.shape[:2]

        lines: list[str] = []
        for b in im.boxes:
            if b.outside:
                continue
            cx = ((b.xtl + b.xbr) / 2) / w
            cy = ((b.ytl + b.ybr) / 2) / h
            bw = (b.xbr - b.xtl) / w
            bh = (b.ybr - b.ytl) / h
            if bw <= 0 or bh <= 0:
                continue
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

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
    LOGGER.info("Wrote %d photo frames+labels for set '%s' -> %s",
                written, set_name, out_raw.resolve())
    LOGGER.info("Next: python projects/price_tag_pipeline/scripts/prepare_data.py "
                "--raw %s --processed data/processed_photos", out_raw)
    return 0 if written else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--xml", type=Path, required=True, help="CVAT export XML")
    p.add_argument("--video", type=Path, help="Source video (video task)")
    p.add_argument("--images-dir", type=Path, help="Folder of source photos (image task)")
    p.add_argument("--set", dest="set_name", help="Photo set id (image task)")
    p.add_argument("--out-raw", type=Path, help="Target raw root (default depends on kind)")
    p.add_argument("--id", dest="video_id", help="video_id (defaults to video stem)")
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args()
    logging.basicConfig(level=getattr(logging, a.log_level), format="%(levelname)s: %(message)s")

    if not a.xml.exists():
        LOGGER.error("XML not found: %s", a.xml)
        return 2

    if a.video:
        out_raw = a.out_raw or Path("data/raw")
        vid = a.video_id or a.video.stem
        return _import_video(a.xml, a.video, out_raw, vid)
    if a.images_dir:
        if not a.set_name:
            LOGGER.error("--set NAME is required with --images-dir")
            return 2
        out_raw = a.out_raw or Path("data/raw_photos")
        return _import_images(a.xml, a.images_dir, a.set_name, out_raw)

    LOGGER.error("Pass --video PATH (video task) or --images-dir DIR --set NAME (photos)")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
