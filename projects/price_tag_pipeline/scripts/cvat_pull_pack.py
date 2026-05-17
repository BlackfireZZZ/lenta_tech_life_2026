#!/usr/bin/env python3
"""Pull validated CVAT annotations → our YOLO format → shareable zip.

Reads tasks straight from the running CVAT (no manual export), converts
each photo's boxes to single-class YOLO, copies the source images, and
packs a self-contained dataset:

    <set>/
      images/<original_name>.jpg
      labels/<original_name>.txt        # YOLO: `0 cx cy w h` (empty = negative)
      classes.txt                       # price_tag
      data.yaml                         # Ultralytics single-class
      manifest.csv                      # scene,image,boxes
      MERGE.md                          # how to fold it in

Then zips it (send to the friend) and, with ``--into``, also drops it into
our pipeline raw layout so ``prepare_data.py`` ingests it unchanged.

    .venv/Scripts/python.exe projects/price_tag_pipeline/scripts/cvat_pull_pack.py \\
        --tasks friends_scene_1,friends_scene_2 \\
        --images-dir friends_labels/dataset_lenta \\
        --user admin --password *** --into data/raw_photos
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import shutil
import sys
import zipfile
from pathlib import Path

THIS = Path(__file__).resolve().parent
SRC = THIS.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from price_tag_pipeline.cv_io import imread, imwrite  # noqa: E402

LOGGER = logging.getLogger("cvat_pull_pack")
DEFAULT_TASKS = "friends_scene_1,friends_scene_2"

MERGE_MD = """\
# {set_name} — validated price-tag boxes (single-class YOLO)

{n_img} images, {n_box} boxes, exported from CVAT (scenes: {scenes}).
Class 0 = `price_tag`. An empty `labels/<name>.txt` = a checked photo with
no tag (a useful negative — keep it).

## Use it

**Ultralytics / any YOLO trainer:** point at `data.yaml` (train=images/).

**This project's pipeline** — drop into the photo raw root and run the
unchanged prepare step:

```
cp images/*  <repo>/data/raw_photos/frames/{set_name}/
cp labels/*  <repo>/data/raw_photos/annotations/labels/{set_name}/
echo price_tag > <repo>/data/raw_photos/annotations/classes.txt
python projects/price_tag_pipeline/scripts/prepare_data.py \\
  --raw data/raw_photos --processed data/processed_photos
```

Or regenerate straight from CVAT next time:
`cvat_pull_pack.py --tasks ... --into data/raw_photos` (idempotent per set).

Filenames are the originals (unique camera timestamps) so re-runs and
multi-source merges never collide.
"""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", default=DEFAULT_TASKS, help="comma list of CVAT task names")
    p.add_argument("--images-dir", type=Path, required=True, help="source photos (match by basename)")
    p.add_argument("--set-name", default="friends_validated")
    p.add_argument("--out", type=Path, default=Path("cvat_exports"))
    p.add_argument("--zip", type=Path, help="zip path (default <out>/<set>.zip)")
    p.add_argument("--labels-only", action="store_true",
                   help="zip excludes images/ (friend already has the photos) — tiny")
    p.add_argument("--into", type=Path, help="also merge into a raw_photos root for prepare_data.py")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--user", default=os.environ.get("CVAT_USER", ""))
    p.add_argument("--password", default=os.environ.get("CVAT_PASSWORD", ""))
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args()
    logging.basicConfig(level=getattr(logging, a.log_level), format="%(levelname)s: %(message)s")
    if not a.user or not a.password:
        LOGGER.error("Need --user/--password (or $CVAT_USER/$CVAT_PASSWORD).")
        return 2
    if not a.images_dir.is_dir():
        LOGGER.error("images dir not found: %s", a.images_dir.resolve())
        return 1

    try:
        from cvat_sdk import make_client
    except ModuleNotFoundError:
        LOGGER.error("cvat-sdk not installed: uv pip install --python .venv cvat-sdk")
        return 2

    want = [s.strip() for s in a.tasks.split(",") if s.strip()]
    root = a.out / a.set_name
    img_dir = root / "images"
    lbl_dir = root / "labels"
    for d in (img_dir, lbl_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    manifest = [("scene", "image", "boxes")]
    n_img = n_box = 0
    scenes_done: list[str] = []

    with make_client(host=a.host, port=a.port, credentials=(a.user, a.password)) as c:
        by_name = {t.name: t for t in c.tasks.list()}
        for name in want:
            task = by_name.get(name)
            if task is None:
                LOGGER.warning("Task %r not found — skipped", name)
                continue
            meta = task.get_meta()
            frames = list(meta.frames)
            ann = task.get_annotations()
            boxes_by_frame: dict[int, list] = {}
            for s in ann.shapes:
                if str(getattr(s.type, "value", s.type)) != "rectangle":
                    continue
                if getattr(s, "outside", False):
                    continue
                boxes_by_frame.setdefault(s.frame, []).append(s.points)

            for fi, fr in enumerate(frames):
                fname = Path(fr.name).name
                src = a.images_dir / fname
                if not src.exists():
                    LOGGER.warning("[%s] source image %s missing — skipped", name, fname)
                    continue
                w, h = fr.width, fr.height
                lines = []
                for pts in boxes_by_frame.get(fi, []):
                    x0, y0, x1, y1 = pts[0], pts[1], pts[2], pts[3]
                    x0, x1 = sorted((max(0.0, min(float(w), x0)), max(0.0, min(float(w), x1))))
                    y0, y1 = sorted((max(0.0, min(float(h), y0)), max(0.0, min(float(h), y1))))
                    if x1 <= x0 or y1 <= y0 or w <= 0 or h <= 0:
                        continue
                    lines.append(
                        f"0 {((x0 + x1) / 2) / w:.6f} {((y0 + y1) / 2) / h:.6f} "
                        f"{(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}"
                    )
                dst_img = img_dir / fname
                if src.suffix.lower() in (".jpg", ".jpeg"):
                    shutil.copy2(src, dst_img)
                else:
                    im = imread(src)
                    if im is None:
                        LOGGER.warning("cannot read %s — skipped", src)
                        continue
                    dst_img = dst_img.with_suffix(".jpg")
                    imwrite(dst_img, im)
                (lbl_dir / f"{dst_img.stem}.txt").write_text(
                    "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
                )
                manifest.append((name, fname, str(len(lines))))
                n_img += 1
                n_box += len(lines)
            scenes_done.append(name)
            LOGGER.info("Pulled %s: %d frames", name, len(frames))

    if not scenes_done:
        LOGGER.error("No tasks pulled — nothing written")
        return 1

    (root / "classes.txt").write_text("price_tag\n", encoding="utf-8")
    (root / "data.yaml").write_text(
        f"path: .\ntrain: images\nval: images\nnc: 1\nnames:\n  0: price_tag\n",
        encoding="utf-8",
    )
    with (root / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(manifest)
    (root / "MERGE.md").write_text(
        MERGE_MD.format(set_name=a.set_name, n_img=n_img, n_box=n_box,
                        scenes=", ".join(scenes_done)),
        encoding="utf-8",
    )

    zip_path = a.zip or (a.out / f"{a.set_name}.zip")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for fp in sorted(root.rglob("*")):
            if not fp.is_file():
                continue
            if a.labels_only and fp.parent.name == "images":
                continue
            z.write(fp, fp.relative_to(root.parent))
    sz = zip_path.stat().st_size / 1e6
    LOGGER.info("Wrote %s (%.1f MB, %s, %d images, %d boxes, scenes=%s)",
                zip_path, sz, "labels-only" if a.labels_only else "full",
                n_img, n_box, ", ".join(scenes_done))

    if a.into:
        fdst = a.into / "frames" / a.set_name
        ldst = a.into / "annotations" / "labels" / a.set_name
        fdst.mkdir(parents=True, exist_ok=True)
        ldst.mkdir(parents=True, exist_ok=True)
        for im in img_dir.iterdir():
            shutil.copy2(im, fdst / im.name)
        for lb in lbl_dir.iterdir():
            shutil.copy2(lb, ldst / lb.name)
        (a.into / "annotations" / "classes.txt").write_text("price_tag\n", encoding="utf-8")
        LOGGER.info("Merged into %s (set=%s). Next: prepare_data.py "
                    "--raw %s --processed data/processed_photos",
                    a.into.resolve(), a.set_name, a.into)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
