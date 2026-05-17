#!/usr/bin/env python3
"""ours → CVAT: turn an already-labelled Lenta CSV into a CVAT seed XML.

Import the produced ``*.cvat.xml`` into a CVAT video task (format
"CVAT 1.1") and the existing boxes appear as tracks — correct the noisy
ones instead of drawing from scratch (briefing §6.1: GT boxes are noisy).

    # one video
    python projects/price_tag_pipeline/scripts/cvat_export_seed.py \\
        --csv real_data/dataset/25_2-10/25_2-10.csv \\
        --video real_data/dataset/25_2-10/25_2-10.mp4 \\
        --out cvat_seeds

    # all five released videos at once
    python projects/price_tag_pipeline/scripts/cvat_export_seed.py \\
        --all --src real_data/dataset --out cvat_seeds
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

from price_tag_pipeline.data.cvat import (  # noqa: E402
    lenta_rows_to_cvat_video_xml,
    read_lenta_csv,
)

LOGGER = logging.getLogger("cvat_export_seed")


def _video_meta(video: Path) -> tuple[int, int, float, int]:
    if cv2 is None:
        raise ModuleNotFoundError("Reading video fps/size needs OpenCV; install opencv-python.")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise OSError(f"Could not open {video}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if w <= 0 or h <= 0:
        raise OSError(f"Bad video dimensions for {video}")
    return w, h, fps, n


def _seed_one(csv_path: Path, video: Path, out_dir: Path) -> Path:
    rows = read_lenta_csv(csv_path)
    w, h, fps, n = _video_meta(video)
    video_id = csv_path.stem
    xml = lenta_rows_to_cvat_video_xml(
        rows, video_id=video_id, width=w, height=h, fps=fps, frame_count=n
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{video_id}.cvat.xml"
    out.write_text(xml, encoding="utf-8")
    LOGGER.info("Wrote %s (%d rows, %dx%d @ %.3ffps, %d frames)",
                out, len(rows), w, h, fps, n)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, help="Single Lenta CSV")
    p.add_argument("--video", type=Path, help="Matching video for --csv")
    p.add_argument("--all", action="store_true", help="Seed every {id}/{id}.csv under --src")
    p.add_argument("--src", type=Path, default=Path("real_data/dataset"))
    p.add_argument("--out", type=Path, default=Path("cvat_seeds"))
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args()
    logging.basicConfig(level=getattr(logging, a.log_level), format="%(levelname)s: %(message)s")

    if a.all:
        if not a.src.is_dir():
            LOGGER.error("Source not found: %s", a.src.resolve())
            return 1
        done = 0
        for entry in sorted(a.src.iterdir()):
            if not entry.is_dir() or entry.name.lower() == "unlabeled":
                continue
            csv_path = entry / f"{entry.name}.csv"
            video = next((entry / f"{entry.name}{e}" for e in (".mp4", ".avi", ".mov", ".mkv")
                          if (entry / f"{entry.name}{e}").exists()), None)
            if not csv_path.exists() or video is None:
                LOGGER.warning("Skipping %s (need %s.csv + video)", entry, entry.name)
                continue
            _seed_one(csv_path, video, a.out)
            done += 1
        LOGGER.info("Seeded %d videos -> %s", done, a.out.resolve())
        return 0 if done else 1

    if not a.csv or not a.video:
        LOGGER.error("Pass --csv and --video, or --all --src DIR")
        return 2
    _seed_one(a.csv, a.video, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
