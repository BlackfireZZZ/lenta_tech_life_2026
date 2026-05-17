#!/usr/bin/env python3
"""Stage the raw hackathon dataset into the canonical ``data/raw`` layout.

The organizers ship the dataset as one folder per labeled video::

    real_data/dataset/
    ├── 25_12-20/{25_12-20.mp4, 25_12-20.csv}
    ├── 25_2-10/{25_2-10.mp4, 25_2-10.csv}
    ├── ...
    ├── sample.csv                 # submission format reference (NOT annotations)
    └── Unlabeled/*.mp4            # videos with no ground truth

The pipeline expects the flattened layout that ``prepare_data.py`` /
``detect_format`` look for::

    data/raw/
    ├── videos/{video_id}.mp4
    ├── unlabeled_videos/*.mp4
    └── annotations/csv/{video_id}.csv   (+ sample.csv, skipped by the loader)

This script bridges the two. It is idempotent (skips files already present
with the same size unless ``--force``) and never mutates the source tree, so
``real_data/`` stays the untouched local source of truth.

The dataset is ~600 MB and is intentionally NOT committed (see ``.gitignore``);
only ``data/splits/*.json`` and docs are versioned. Re-run this after a fresh
clone / new worktree to repopulate ``data/raw``.

Usage (run from the repo that will train/infer — usually the main checkout)::

    python projects/price_tag_pipeline/scripts/ingest_real_data.py \
        --src real_data/dataset --dst data/raw

    # space-saving on the same volume (Windows/NTFS, Linux, macOS):
    python projects/price_tag_pipeline/scripts/ingest_real_data.py --mode hardlink
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

LOGGER = logging.getLogger("ingest_real_data")

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")
SAMPLE_CSV_NAME = "sample.csv"


def _same_size(a: Path, b: Path) -> bool:
    return a.exists() and b.exists() and a.stat().st_size == b.stat().st_size


def _place(src: Path, dst: Path, mode: str, force: bool) -> str:
    """Place ``src`` at ``dst`` using copy/hardlink/move. Returns an action tag."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not force:
        if _same_size(src, dst):
            return "skip"
        LOGGER.warning("Overwriting %s (size differs from source)", dst)
    if dst.exists():
        dst.unlink()

    if mode == "copy":
        shutil.copy2(src, dst)
        return "copy"
    if mode == "move":
        shutil.move(str(src), str(dst))
        return "move"
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError as exc:  # cross-volume or unsupported FS -> fall back
            LOGGER.warning("Hardlink failed (%s); copying instead", exc)
            shutil.copy2(src, dst)
            return "copy"
    raise ValueError(f"Unknown mode: {mode}")


def _find_video(folder: Path, stem: str) -> Path | None:
    for ext in VIDEO_EXTS:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default="real_data/dataset", help="Source dataset folder (organizer layout)")
    p.add_argument("--dst", default="data/raw", help="Destination data/raw root")
    p.add_argument(
        "--mode",
        default="copy",
        choices=["copy", "hardlink", "move"],
        help="copy (safe, default) | hardlink (no extra space, same volume) | move",
    )
    p.add_argument("--force", action="store_true", help="Re-place files even if a same-size copy exists")
    p.add_argument("--no-unlabeled", action="store_true", help="Skip the Unlabeled/ videos")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    src = Path(args.src).expanduser()
    dst = Path(args.dst).expanduser()
    if not src.is_dir():
        LOGGER.error("Source not found: %s (pass --src; the dataset lives in real_data/dataset)", src.resolve())
        return 1

    videos_dir = dst / "videos"
    unlabeled_dir = dst / "unlabeled_videos"
    csv_dir = dst / "annotations" / "csv"

    stats: dict[str, int] = {"video": 0, "csv": 0, "unlabeled": 0, "skip": 0}
    pairs: list[str] = []

    for entry in sorted(src.iterdir()):
        if not entry.is_dir() or entry.name.lower() == "unlabeled":
            continue
        video_id = entry.name
        video = _find_video(entry, video_id)
        csv_path = entry / f"{video_id}.csv"
        if video is None or not csv_path.exists():
            LOGGER.warning("Skipping %s: expected %s.mp4 + %s.csv", entry, video_id, video_id)
            continue

        a1 = _place(video, videos_dir / video.name, args.mode, args.force)
        a2 = _place(csv_path, csv_dir / csv_path.name, args.mode, args.force)
        stats["video"] += a1 != "skip"
        stats["csv"] += a2 != "skip"
        stats["skip"] += (a1 == "skip") + (a2 == "skip")
        pairs.append(video_id)

    # Submission-format reference. The loader skips it by name; keep it next to
    # the real CSVs so the expected output schema is discoverable.
    sample = src / SAMPLE_CSV_NAME
    if sample.exists():
        _place(sample, csv_dir / SAMPLE_CSV_NAME, args.mode, args.force)

    if not args.no_unlabeled:
        unlabeled_src = src / "Unlabeled"
        if unlabeled_src.is_dir():
            for vid in sorted(unlabeled_src.iterdir()):
                if vid.suffix.lower() in VIDEO_EXTS:
                    a = _place(vid, unlabeled_dir / vid.name, args.mode, args.force)
                    stats["unlabeled"] += a != "skip"
                    stats["skip"] += a == "skip"

    LOGGER.info("Labeled videos staged: %s", ", ".join(pairs) or "(none)")
    LOGGER.info(
        "Done. videos=%d csv=%d unlabeled=%d skipped(existing)=%d -> %s",
        stats["video"],
        stats["csv"],
        stats["unlabeled"],
        stats["skip"],
        dst.resolve(),
    )
    LOGGER.info("Next: python projects/price_tag_pipeline/scripts/prepare_data.py --raw %s", dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
