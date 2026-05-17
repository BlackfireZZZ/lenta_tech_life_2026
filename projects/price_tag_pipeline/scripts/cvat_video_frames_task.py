#!/usr/bin/env python3
"""Slice videos into every-N-seconds frames → CVAT image-task seeds.

A different take on the released videos: instead of an interpolation task
(which smears the organizers' awful boxes across frames), sample one frame
every ``--every`` seconds and make a plain **image task** — box them
yourself, detector-first. The organizer annotation is kept **only as a
single-frame hint per tag**: each released tag's box is placed on the one
sampled frame nearest its timestamp, never propagated ("далее она ужасна").

Emits the same ``manifest.json`` schema ``cvat_bootstrap_photos.py`` eats,
so the existing bootstrap creates the image tasks + imports the seeds:

    .venv/Scripts/python.exe projects/price_tag_pipeline/scripts/cvat_video_frames_task.py \\
        --all --src "E:/Hackatons/lenta_tech_life_2026/real_data/dataset" \\
        --every 2.0 --out cvat_seeds_frames --frames-dir cvat_frames
    .venv/Scripts/python.exe projects/price_tag_pipeline/scripts/cvat_bootstrap_photos.py \\
        --manifest cvat_seeds_frames/manifest.json --user admin --password ***
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

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None  # type: ignore[assignment]

from price_tag_pipeline.cv_io import imwrite  # noqa: E402
from price_tag_pipeline.data.cvat import (  # noqa: E402
    ImageBoxes,
    image_boxes_to_cvat_images_xml,
    read_lenta_csv,
)
from price_tag_pipeline.data.loaders import (  # noqa: E402
    _parse_decimal,
    _parse_lenta_frame_idx,
    _read_frame_near,
)

LOGGER = logging.getLogger("cvat_video_frames_task")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


def _find_video(folder: Path, stem: str) -> Path | None:
    return next((folder / f"{stem}{e}" for e in VIDEO_EXTS
                 if (folder / f"{stem}{e}").exists()), None)


def _bbox(row: dict, w: int, h: int):
    vals = [_parse_decimal(row.get(k)) for k in ("x_min", "y_min", "x_max", "y_max")]
    if any(v is None for v in vals):
        return None
    x0, y0, x1, y1 = (float(v) for v in vals)  # type: ignore[arg-type]
    x0, x1 = (max(0.0, min(float(w), v)) for v in (x0, x1))
    y0, y1 = (max(0.0, min(float(h), v)) for v in (y0, y1))
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def _process(video: Path, csv_path: Path, every: float,
             frames_root: Path, seeds_root: Path) -> dict | None:
    if cv2 is None:
        raise ModuleNotFoundError("Needs OpenCV; install opencv-python.")
    vid = csv_path.stem
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        LOGGER.warning("cannot open %s", video)
        return None
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if n <= 0 or w <= 0 or h <= 0:
        cap.release()
        LOGGER.warning("bad video meta for %s", video)
        return None

    step = max(1, int(round(every * fps)))
    sampled = list(range(0, n, step))
    fdir = frames_root / vid
    fdir.mkdir(parents=True, exist_ok=True)
    written: list[tuple[int, str]] = []  # (frame_idx, filename)
    for idx in sampled:
        fn = f"{idx:06d}.jpg"
        dst = fdir / fn
        if not dst.exists():
            frame = _read_frame_near(cap, idx, frame_count=n)
            if frame is None:
                continue
            imwrite(dst, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        written.append((idx, fn))
    cap.release()
    if not written:
        LOGGER.warning("no frames sampled for %s", vid)
        return None

    sampled_idxs = [i for i, _ in written]
    name_of = {i: fn for i, fn in written}

    # Each organizer tag → its box on the single nearest sampled frame.
    per_frame: dict[int, ImageBoxes] = {}
    n_box = 0
    for row in read_lenta_csv(csv_path):
        f_tag = _parse_lenta_frame_idx(row.get("frame_timestamp"), fps=fps, frame_count=n)
        box = _bbox(row, w, h)
        if f_tag is None or box is None:
            continue
        nearest = min(sampled_idxs, key=lambda s: abs(s - f_tag))
        rec = per_frame.setdefault(
            nearest, ImageBoxes(name=name_of[nearest], width=w, height=h)
        )
        rec.boxes.append(box)
        n_box += 1

    seeds_root.mkdir(parents=True, exist_ok=True)
    xml_path = seeds_root / f"{vid}.cvat.xml"
    xml_path.write_text(
        image_boxes_to_cvat_images_xml(
            [per_frame[k] for k in sorted(per_frame)], task_name=vid
        ),
        encoding="utf-8",
    )
    LOGGER.info("%s: %d frames @ every %.1fs (step %d), %d organizer boxes on %d frames",
                vid, len(written), every, step, n_box, len(per_frame))
    return {
        "scene": vid,
        "xml": str(xml_path),
        "images": [str((fdir / fn).resolve()) for _, fn in written],
        "total_images": len(written),
        "boxed_images": len(per_frame),
        "boxes": n_box,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, default=Path("real_data/dataset"))
    p.add_argument("--all", action="store_true", help="every {id}/{id}.csv+video under --src")
    p.add_argument("--csv", type=Path, help="single Lenta CSV (with --video)")
    p.add_argument("--video", type=Path, help="matching video for --csv")
    p.add_argument("--every", type=float, default=2.0, help="seconds between sampled frames")
    p.add_argument("--out", type=Path, default=Path("cvat_seeds_frames"))
    p.add_argument("--frames-dir", type=Path, default=Path("cvat_frames"))
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args()
    logging.basicConfig(level=getattr(logging, a.log_level), format="%(levelname)s: %(message)s")

    jobs: list[tuple[Path, Path]] = []
    if a.all:
        if not a.src.is_dir():
            LOGGER.error("src not found: %s", a.src.resolve())
            return 1
        for entry in sorted(a.src.iterdir()):
            if not entry.is_dir() or entry.name.lower() == "unlabeled":
                continue
            cp = entry / f"{entry.name}.csv"
            vp = _find_video(entry, entry.name)
            if cp.exists() and vp is not None:
                jobs.append((vp, cp))
    elif a.csv and a.video:
        jobs.append((a.video, a.csv))
    else:
        LOGGER.error("Pass --all --src DIR, or --csv CSV --video VIDEO")
        return 2

    manifest = []
    for video, csv_path in jobs:
        entry = _process(video, csv_path, a.every, a.frames_dir, a.out)
        if entry:
            manifest.append(entry)
    if not manifest:
        LOGGER.error("nothing produced")
        return 1
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tot_i = sum(m["total_images"] for m in manifest)
    tot_b = sum(m["boxes"] for m in manifest)
    LOGGER.info("Done. %d videos, %d frames, %d organizer hint-boxes -> %s",
                len(manifest), tot_i, tot_b, (a.out / "manifest.json"))
    LOGGER.info("Next: cvat_bootstrap_photos.py --manifest %s --user admin --password ***",
                a.out / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
