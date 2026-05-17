#!/usr/bin/env python3
"""External photo dataset → CVAT-images seed XMLs, split into N scenes.

Handles a friend's detector dump where the labels are a candidates CSV
(``all_candidates.csv`` with ``candidate_type``/absolute+YOLO coords) — not
classic YOLO ``.txt`` — and also plain YOLO-txt via ``--yolo-labels``.

Every photo (boxed or not) is assigned to one of ``--scenes`` chronological
chunks so you validate/extend in manageable CVAT image tasks. Only photos
with boxes appear in the XML; the rest ride along in the task as blanks to
label. A ``manifest.json`` tells the bootstrap which files to upload where.

    python projects/price_tag_pipeline/scripts/cvat_from_external.py \\
        --csv friends_labels/all_candidates.csv \\
        --images-dir friends_labels/dataset_lenta \\
        --scenes 4 --out cvat_seeds_friends
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
SRC = THIS.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from price_tag_pipeline.data.cvat import (  # noqa: E402
    MODEL_BACKED_SOURCES,
    ImageBoxes,
    candidates_csv_to_image_boxes,
    image_boxes_to_cvat_images_xml,
    read_lenta_csv,
    yolo_txt_to_image_boxes,
)

LOGGER = logging.getLogger("cvat_from_external")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def _chunks(items: list, n: int) -> list[list]:
    n = max(1, min(n, len(items)))
    k, m = divmod(len(items), n)
    out, i = [], 0
    for s in range(n):
        size = k + (1 if s < m else 0)
        out.append(items[i:i + size])
        i += size
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images-dir", type=Path, required=True)
    p.add_argument("--csv", type=Path, help="all_candidates.csv (CSV mode)")
    p.add_argument("--yolo-labels", type=Path, help="dir of YOLO <stem>.txt (YOLO mode)")
    p.add_argument("--candidate-type", default="merged_final",
                   help="comma list kept from CSV (default: merged_final)")
    p.add_argument("--model-backed", action="store_true",
                   help="drop classic_color_cv noise — keep only YOLO-backed boxes")
    p.add_argument("--sources", help="explicit comma list of `source` values to keep")
    p.add_argument("--min-conf", type=float, default=0.0)
    p.add_argument("--scenes", type=int, default=4)
    p.add_argument("--out", type=Path, default=Path("cvat_seeds_friends"))
    p.add_argument("--prefix", default="friends_scene")
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args()
    logging.basicConfig(level=getattr(logging, a.log_level), format="%(levelname)s: %(message)s")

    if not a.images_dir.is_dir():
        LOGGER.error("images dir not found: %s", a.images_dir.resolve())
        return 1
    images = sorted(f for f in a.images_dir.iterdir() if f.suffix.lower() in IMG_EXTS)
    if not images:
        LOGGER.error("No images under %s", a.images_dir)
        return 1

    if a.csv:
        rows = read_lenta_csv(a.csv)
        if a.sources:
            sources: set[str] | None = {s.strip() for s in a.sources.split(",")}
        elif a.model_backed:
            sources = MODEL_BACKED_SOURCES
        else:
            sources = None
        boxes = candidates_csv_to_image_boxes(
            rows,
            candidate_types={s.strip() for s in a.candidate_type.split(",")},
            min_conf=a.min_conf,
            sources=sources,
        )
        LOGGER.info("CSV %d rows -> %d images with boxes (type=%s, sources=%s)",
                    len(rows), len(boxes), a.candidate_type,
                    "model-backed" if sources else "all")
    elif a.yolo_labels:
        boxes = yolo_txt_to_image_boxes(a.images_dir, a.yolo_labels)
        LOGGER.info("YOLO-txt -> %d images with boxes", len(boxes))
    else:
        LOGGER.error("Pass --csv or --yolo-labels")
        return 2

    a.out.mkdir(parents=True, exist_ok=True)
    scenes = _chunks(images, a.scenes)
    manifest = []
    total_boxed = total_box = 0
    for i, scene in enumerate(scenes, 1):
        boxed: list[ImageBoxes] = [boxes[f.name] for f in scene if f.name in boxes]
        name = f"{a.prefix}_{i}"
        xml_path = a.out / f"{name}.cvat.xml"
        xml_path.write_text(
            image_boxes_to_cvat_images_xml(boxed, task_name=name), encoding="utf-8"
        )
        nb = sum(len(b.boxes) for b in boxed)
        total_boxed += len(boxed)
        total_box += nb
        manifest.append({
            "scene": name,
            "xml": str(xml_path),
            "images": [str(f.resolve()) for f in scene],
            "total_images": len(scene),
            "boxed_images": len(boxed),
            "boxes": nb,
        })
        LOGGER.info("%s: %d images (%d boxed, %d boxes) -> %s",
                    name, len(scene), len(boxed), nb, xml_path.name)

    (a.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOGGER.info("Done. %d scenes, %d/%d images boxed, %d boxes total -> %s",
                len(scenes), total_boxed, len(images), total_box, a.out.resolve())
    LOGGER.info("Next: cvat_bootstrap_photos.py --manifest %s --user admin --password ***",
                a.out / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
