#!/usr/bin/env python3
"""STEP 1 — slice organizer videos into frames (every N s) + drop duplicates.

Runs on the **laptop** (CPU only — just OpenCV).  For each input video it
writes one JPEG every ``--every`` seconds into
``<out>/<scene>/NNNNNN.jpg`` and skips frames where the robot was parked
(near-duplicate of the last kept frame — see ``data.frame_sampling``).

Output layout (Ultralytics-style, label files land next to images in STEP 2):

    cvat_video_frames/
      <scene>/000000.jpg 000048.jpg ...
      slice_manifest.json          # {scene: [frame files]} + counts

Next: STEP 2 ``prelabel_frames.py`` (local or Google Colab).

    .venv/Scripts/python.exe projects/price_tag_pipeline/scripts/slice_video_frames.py \\
        --src "E:/Hackatons/lenta_tech_life_2026/real_data/dataset" \\
        --every 2.0 --out cvat_video_frames
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

from price_tag_pipeline.cv_io import imwrite  # noqa: E402
from price_tag_pipeline.data.frame_sampling import (  # noqa: E402
    DEFAULT_MIN_DIFF,
    iter_sampled_frames,
)

LOGGER = logging.getLogger("slice_video_frames")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".MP4", ".MOV")


def _discover(src: Path, include_unlabeled: bool) -> list[Path]:
    """Every video under ``src`` (recursive), sorted, stable scene order.

    A ``Unlabeled/`` (case-insensitive) subtree is skipped by default — the
    organizer set keeps held-out clips there; mixing them in silently would
    double the work under auto-disambiguated scene names. Opt in with
    ``--include-unlabeled``.
    """
    out = []
    for p in src.rglob("*"):
        if not (p.is_file() and p.suffix in VIDEO_EXTS):
            continue
        if not include_unlabeled and any(
            part.lower() == "unlabeled" for part in p.relative_to(src).parts
        ):
            continue
        out.append(p)
    return sorted(out)


def _scene_name(video: Path, taken: set[str]) -> str:
    # real_data/dataset/<id>/<id>.mp4 -> "<id>"; flat folders -> stem.
    name = video.stem
    if name in taken:  # disambiguate identical stems across folders
        name = f"{video.parent.name}__{video.stem}"
    i = 2
    base = name
    while name in taken:
        name = f"{base}_{i}"
        i += 1
    taken.add(name)
    return name


def _slice_one(video: Path, scene: str, every: float, min_diff: float,
               out_root: Path, jpeg_q: int) -> dict | None:
    sdir = out_root / scene
    sdir.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    try:
        import cv2  # noqa: F401  (only for the JPEG quality flag)
        from cv2 import IMWRITE_JPEG_QUALITY
        jpeg_params = [IMWRITE_JPEG_QUALITY, jpeg_q]
    except Exception:
        jpeg_params = None

    seen = sampled = 0
    for idx, frame, total, fps in iter_sampled_frames(
        video, every_s=every, min_diff=min_diff
    ):
        sampled += 1
        fn = f"{idx:06d}.jpg"
        dst = sdir / fn
        if not dst.exists():
            imwrite(dst, frame, jpeg_params)
        files.append(fn)
        seen = total
    if not files:
        LOGGER.warning("%s: no frames produced", scene)
        return None
    LOGGER.info("%s: kept %d frames (every %.1fs, dedup<%.3f) <- %s",
                scene, len(files), every, min_diff, video.name)
    return {
        "scene": scene,
        "video": str(video),
        "dir": str(sdir.resolve()),
        "frames": files,
        "kept": len(files),
        "source_total": seen,
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--src", type=Path, help="folder of videos (recursed)")
    g.add_argument("--video", type=Path, help="a single video file")
    p.add_argument("--every", type=float, default=2.0,
                   help="seconds between sampled frames (default 2)")
    p.add_argument("--min-diff", type=float, default=DEFAULT_MIN_DIFF,
                   help="drop a frame within this normalized gray-diff of the "
                        "last kept one (robot-parked dedup). 0 = keep all")
    p.add_argument("--out", type=Path, default=Path("cvat_video_frames"))
    p.add_argument("--include-unlabeled", action="store_true",
                   help="also slice an Unlabeled/ subtree (skipped by default)")
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args()
    logging.basicConfig(level=getattr(logging, a.log_level),
                        format="%(levelname)s: %(message)s")

    if a.video:
        videos = [a.video] if a.video.is_file() else []
    else:
        if not a.src.is_dir():
            LOGGER.error("src not found: %s", a.src.resolve())
            return 1
        videos = _discover(a.src, a.include_unlabeled)
    if not videos:
        LOGGER.error("no videos found")
        return 1

    a.out.mkdir(parents=True, exist_ok=True)
    taken: set[str] = set()
    manifest = []
    for v in videos:
        entry = _slice_one(v, _scene_name(v, taken), a.every, a.min_diff,
                            a.out, a.jpeg_quality)
        if entry:
            manifest.append(entry)
    if not manifest:
        LOGGER.error("nothing produced")
        return 1

    (a.out / "slice_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tot = sum(m["kept"] for m in manifest)
    LOGGER.info("Done. %d videos -> %d frames in %s",
                len(manifest), tot, a.out.resolve())
    LOGGER.info("Next (STEP 2): prelabel_frames.py --frames-dir %s "
                "(run on Colab if no GPU)", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
