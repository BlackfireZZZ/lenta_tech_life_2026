#!/usr/bin/env python3
"""STEP 3 — pre-labelled frames → CVAT image-task seeds + bootstrap manifest.

Runs on the **laptop**.  Reads the ``slice_video_frames.py`` tree (each
``<scene>/`` holds the JPEGs and, after STEP 2, a colocated YOLO ``<stem>.txt``)
and emits one *CVAT for images 1.1* seed XML per scene plus the
``manifest.json`` that ``cvat_bootstrap_photos.py`` consumes — same schema
as the friend-scene flow, so push is the existing one-liner.

Every sampled frame is uploaded (empty ones too — scanning a blank frame and
confirming "no tag" is a useful negative); the detector's boxes are
pre-loaded so the friend **validates/corrects** instead of drawing from
scratch.

    .venv/Scripts/python.exe projects/price_tag_pipeline/scripts/frames_to_cvat.py \\
        --frames-dir cvat_video_frames --out cvat_seeds_video
    .venv/Scripts/python.exe projects/price_tag_pipeline/scripts/cvat_bootstrap_photos.py \\
        --manifest cvat_seeds_video/manifest.json --user admin --password ***
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
    image_boxes_to_cvat_images_xml,
    yolo_txt_to_image_boxes,
)

LOGGER = logging.getLogger("frames_to_cvat")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def _scene_dirs(root: Path) -> list[Path]:
    subs = [d for d in sorted(root.iterdir())
            if d.is_dir() and any(p.suffix.lower() in IMG_EXTS
                                  for p in d.iterdir() if p.is_file())]
    if subs:
        return subs
    if any(p.suffix.lower() in IMG_EXTS for p in root.iterdir() if p.is_file()):
        return [root]
    return []


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--frames-dir", type=Path, default=Path("cvat_video_frames"))
    p.add_argument("--out", type=Path, default=Path("cvat_seeds_video"))
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args()
    logging.basicConfig(level=getattr(logging, a.log_level),
                        format="%(levelname)s: %(message)s")

    if not a.frames_dir.is_dir():
        LOGGER.error("frames dir not found: %s", a.frames_dir.resolve())
        return 1
    scenes = _scene_dirs(a.frames_dir)
    if not scenes:
        LOGGER.error("no scene folders with images under %s",
                     a.frames_dir.resolve())
        return 1
    a.out.mkdir(parents=True, exist_ok=True)

    manifest = []
    total_unlabelled = 0
    for sdir in scenes:
        imgs = sorted(p for p in sdir.iterdir()
                      if p.is_file() and p.suffix.lower() in IMG_EXTS)
        if not imgs:
            continue
        # colocated labels (labels_dir defaults to images_dir); only frames
        # with a non-empty .txt come back as ImageBoxes.
        boxed = yolo_txt_to_image_boxes(sdir)
        if not boxed:
            total_unlabelled += 1
        recs = [boxed[k] for k in sorted(boxed)]
        xml_path = a.out / f"{sdir.name}.cvat.xml"
        xml_path.write_text(
            image_boxes_to_cvat_images_xml(recs, task_name=sdir.name),
            encoding="utf-8",
        )
        n_box = sum(len(r.boxes) for r in recs)
        LOGGER.info("%s: %d frames, %d pre-boxed, %d boxes -> %s",
                    sdir.name, len(imgs), len(recs), n_box, xml_path.name)
        manifest.append({
            "scene": sdir.name,
            "xml": str(xml_path),
            "images": [str(p.resolve()) for p in imgs],
            "total_images": len(imgs),
            "boxed_images": len(recs),
            "boxes": n_box,
        })

    if not manifest:
        LOGGER.error("nothing produced")
        return 1
    (a.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tot_i = sum(m["total_images"] for m in manifest)
    tot_b = sum(m["boxes"] for m in manifest)
    if total_unlabelled:
        LOGGER.warning("%d scene(s) had no labels — did STEP 2 "
                       "(prelabel_frames.py) run? Frames upload blank.",
                       total_unlabelled)
    LOGGER.info("Done. %d scenes, %d frames, %d seed boxes -> %s",
                len(manifest), tot_i, tot_b, (a.out / "manifest.json"))
    LOGGER.info("Next: cvat_bootstrap_photos.py --manifest %s "
                "--user admin --password ***", a.out / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
