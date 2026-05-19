#!/usr/bin/env python3
"""Turn outputs/shelf_audit/<video>/ into a static frontend fixture.

The backend/ml is a mocked skeleton, so the Shelf Audit UI page reads a static
fixture served from Vite's ``public/`` (zero-backend jury demo). This copies
``audit.json`` and rotates the card/evidence crops **upright** for display
(the robot cam is mounted 90° CW, so crops cut from the original frame are
sideways — rotate CCW to view normally), then writes an ``index.json`` the
page uses as a video selector.

Pure tooling: reads outputs/, writes frontend/public/. Touches nothing graded.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import sys

import cv2

# Unicode-safe still-image I/O — cv2.imread/imwrite silently fail (None/False)
# on non-ASCII paths, and the build machine sits under a Cyrillic Windows
# username. Resolve the pipeline package the same way the other scripts do.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from price_tag_pipeline.cv_io import imread as cv_imread, imwrite as cv_imwrite  # noqa: E402

MAIN_TREE = Path("E:/Hackatons/lenta_tech_life_2026")
DEFAULT_SRC = MAIN_TREE / "outputs/shelf_audit"
DEFAULT_DST = (MAIN_TREE / ".claude/worktrees/shelf-analytics"
               / "frontend/public/shelf-audit")


def _rotate_upright(src: Path, dst: Path) -> bool:
    img = cv_imread(str(src))
    if img is None:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Robot cam is 90° CW → rotate CCW to display upright.
    return cv_imwrite(str(dst), cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))


def _copy_video(video_dir: Path, dst_root: Path) -> dict | None:
    audit_path = video_dir / "audit.json"
    if not audit_path.exists():
        return None
    vid = video_dir.name
    out = dst_root / vid
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    (out / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    n_img = 0
    for sub in ("cards", "crops"):
        sdir = video_dir / sub
        if not sdir.is_dir():
            continue
        for img in sorted(sdir.glob("*.jpg")):
            if _rotate_upright(img, out / sub / img.name):
                n_img += 1

    s = audit.get("summary", {})
    print(f"  {vid}: {len(audit.get('alerts', []))} alerts, "
          f"{len(audit.get('cards', []))} cards, {n_img} imgs")
    return {
        "id": vid,
        "n_alerts": len(audit.get("alerts", [])),
        "n_cards": len(audit.get("cards", [])),
        "n_out_of_stock": s.get("n_out_of_stock", 0),
        "n_missing_price_tag": s.get("n_missing_price_tag", 0),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, default=DEFAULT_SRC)
    p.add_argument("--dst", type=Path, default=DEFAULT_DST)
    args = p.parse_args()

    if not args.src.is_dir():
        raise SystemExit(f"No shelf-audit outputs at {args.src} — run run_shelf_audit.py first")

    args.dst.mkdir(parents=True, exist_ok=True)
    print(f"[fixture] {args.src} -> {args.dst}")
    videos = []
    for vdir in sorted(d for d in args.src.iterdir() if d.is_dir() and not d.name.startswith("_")):
        info = _copy_video(vdir, args.dst)
        if info:
            videos.append(info)
    (args.dst / "index.json").write_text(
        json.dumps({"videos": videos}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"[fixture] {len(videos)} video(s) -> index.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
