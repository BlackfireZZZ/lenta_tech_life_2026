#!/usr/bin/env python3
"""Prepare a CVAT task: emit the label spec + print exact setup steps.

Writes ``cvat_label_spec.json`` (paste into CVAT → Constructor → "Raw", or
import via the label JSON) so every task uses the *same* ``price_tag``
schema the importer expects. Optionally pre-rotates a source video/photo
folder (OFF by default — see the runbook for why we annotate in the
original, un-rotated orientation to stay coordinate-compatible with the
released CSVs and the training pipeline).

    python projects/price_tag_pipeline/scripts/cvat_prepare_task.py --label-spec
    python projects/price_tag_pipeline/scripts/cvat_prepare_task.py \\
        --video real_data/dataset/49_5/49_5.mp4
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

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None  # type: ignore[assignment]

from price_tag_pipeline.cv_io import imread, imwrite  # noqa: E402
from price_tag_pipeline.data.cvat import label_spec_json  # noqa: E402

LOGGER = logging.getLogger("cvat_prepare_task")

_STEPS = """\
CVAT task setup
===============
1. Open http://localhost:8080 and log in.
2. Projects → + → "Lenta price tags". Open it → Constructor → switch to
   "Raw" → paste the contents of {spec} → Done. (All tasks in this project
   then inherit the price_tag label + attributes the importer expects.)
3. Tasks → + :
     • VIDEO: Name = the video_id (e.g. 25_2-10). Upload the .mp4.
       Leave "Use cache" on. Submit. This is an interpolation task —
       tracking works (see the runbook's "Tracking in CVAT" section).
     • PHOTOS: Name = a set id (e.g. store_run_1). Upload the images as
       "Image" data. (No tracking — each photo is independent.)
4. (Video only, to start from existing labels) open the task → Actions →
   "Upload annotations" → format "CVAT 1.1" → pick the matching
   cvat_seeds/<video_id>.cvat.xml produced by cvat_export_seed.py.
5. Annotate / correct. Export: task → Actions → "Export task dataset" →
   format "CVAT for video 1.1" (video) or "CVAT for images 1.1" (photos),
   "Save images" OFF → download the .zip, unzip → annotations.xml.
6. Hand the XML back; cvat_import.py folds it into data/raw and
   prepare_data.py extends the dataset — no pipeline change.
"""


def _rotate_video(src: Path, dst: Path, cw: bool) -> None:
    if cv2 is None:
        raise ModuleNotFoundError("Rotation needs OpenCV; install opencv-python.")
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise OSError(f"Could not open {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    code = cv2.ROTATE_90_CLOCKWISE if cw else cv2.ROTATE_90_COUNTERCLOCKWISE
    out_w, out_h = (h, w)
    vw = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        vw.write(cv2.rotate(frame, code))
    cap.release()
    vw.release()
    LOGGER.info("Rotated %s -> %s (%dx%d)", src.name, dst, out_w, out_h)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label-spec", action="store_true", help="Write cvat_label_spec.json and print it")
    p.add_argument("--out", type=Path, default=Path("cvat_label_spec.json"))
    p.add_argument("--video", type=Path, help="Source video to prepare")
    p.add_argument("--photos", type=Path, help="Folder of source photos")
    p.add_argument("--rotate", choices=["cw", "ccw"], help="Pre-rotate 90° (default: none)")
    p.add_argument("--staging", type=Path, default=Path("cvat_staging"))
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args()
    logging.basicConfig(level=getattr(logging, a.log_level), format="%(levelname)s: %(message)s")

    spec = label_spec_json()
    a.out.write_text(spec, encoding="utf-8")
    LOGGER.info("Wrote label spec -> %s", a.out.resolve())

    if a.video and a.rotate:
        a.staging.mkdir(parents=True, exist_ok=True)
        _rotate_video(a.video, a.staging / a.video.name, cw=a.rotate == "cw")
    elif a.photos and a.rotate:
        a.staging.mkdir(parents=True, exist_ok=True)
        code = cv2.ROTATE_90_CLOCKWISE if a.rotate == "cw" else cv2.ROTATE_90_COUNTERCLOCKWISE
        for img_path in sorted(a.photos.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            img = imread(img_path)
            if img is None:
                continue
            imwrite(a.staging / img_path.name, cv2.rotate(img, code))
        LOGGER.info("Rotated photos -> %s", a.staging.resolve())

    print()
    print(_STEPS.format(spec=a.out))
    if a.label_spec:
        print("----- cvat_label_spec.json -----")
        print(spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
