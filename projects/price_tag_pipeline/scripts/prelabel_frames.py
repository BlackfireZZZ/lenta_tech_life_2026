#!/usr/bin/env python3
"""STEP 2 — pre-annotate sliced frames with our trained detector.

"Our detector" = the OpenFoodFacts price-tag YOLO the friend's notebook
uses: ``hf://openfoodfacts/price-tag-detection/weights/best.pt`` (the fixed
base of our solution; override with ``--model`` to use a fine-tuned
checkpoint, e.g. ``data/checkpoints/detector/best.pt``).  Recall-first
defaults mirror that notebook (conf 0.05, iou 0.50, imgsz 1280) — better to
delete a wrong box in CVAT than to hand-draw a missed one.

GPU-heavy → **run this on Google Colab** if the laptop has no GPU
(``notebooks/lenta_prelabel_colab.ipynb`` is this exact step, click-by-click).
Locally it needs ``ultralytics`` + ``huggingface-hub`` in the venv.

Input/output: a ``slice_video_frames.py`` tree.  A YOLO ``<stem>.txt`` is
written **next to each image** (empty file = checked, no tag — a useful
negative). Then STEP 3 ``frames_to_cvat.py``.

    .venv/Scripts/python.exe projects/price_tag_pipeline/scripts/prelabel_frames.py \\
        --frames-dir cvat_video_frames
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

from price_tag_pipeline.detector import resolve_detector_model_path  # noqa: E402

LOGGER = logging.getLogger("prelabel_frames")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
DEFAULT_MODEL = "hf://openfoodfacts/price-tag-detection/weights/best.pt"


def _scene_dirs(root: Path) -> list[Path]:
    """Scene subfolders with images; fall back to ``root`` itself if flat."""
    subs = [d for d in sorted(root.iterdir())
            if d.is_dir() and any(p.suffix.lower() in IMG_EXTS
                                  for p in d.iterdir() if p.is_file())]
    if subs:
        return subs
    if any(p.suffix.lower() in IMG_EXTS for p in root.iterdir() if p.is_file()):
        return [root]
    return []


def prelabel_dir(
    frames_dir: Path,
    *,
    model: str = DEFAULT_MODEL,
    conf: float = 0.05,
    iou: float = 0.50,
    imgsz: int = 1280,
    device: str | None = None,
) -> list[dict]:
    """Detector → colocated YOLO ``.txt``.  Returns a per-scene summary.

    Importable so the Colab notebook reuses the exact same logic.
    """
    try:
        from ultralytics import YOLO  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ultralytics not installed. Either run STEP 2 on Google Colab "
            "(notebooks/lenta_prelabel_colab.ipynb), or locally: "
            "uv pip install --python .venv ultralytics huggingface-hub"
        ) from exc

    weights = resolve_detector_model_path(model)
    LOGGER.info("Loading detector: %s", weights)
    net = YOLO(weights)

    summary: list[dict] = []
    for sdir in _scene_dirs(frames_dir):
        imgs = sorted(p for p in sdir.iterdir()
                      if p.is_file() and p.suffix.lower() in IMG_EXTS)
        if not imgs:
            continue
        n_box = 0
        boxed = 0
        # stream=True: bounded memory over a long scene.
        results = net.predict(
            source=[str(p) for p in imgs], stream=True, verbose=False,
            conf=conf, iou=iou, imgsz=imgsz, device=device,
        )
        for img_path, res in zip(imgs, results):
            h, w = res.orig_shape  # (H, W)
            lines: list[str] = []
            boxes = getattr(res, "boxes", None)
            if boxes is not None and len(boxes):
                for xywhn in boxes.xywhn:  # already normalized cx,cy,w,h
                    cx, cy, bw, bh = (float(v) for v in xywhn.tolist())
                    if bw <= 0 or bh <= 0:
                        continue
                    lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            (img_path.with_suffix(".txt")).write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
            n_box += len(lines)
            boxed += 1 if lines else 0
        LOGGER.info("%s: %d images, %d boxes (%d boxed)",
                    sdir.name, len(imgs), n_box, boxed)
        summary.append({
            "scene": sdir.name, "images": len(imgs),
            "boxed_images": boxed, "boxes": n_box,
        })
    return summary


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--frames-dir", type=Path, default=Path("cvat_video_frames"))
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="hf:// URI, local .pt, or Ultralytics name "
                        "(default: the OpenFoodFacts price-tag detector)")
    p.add_argument("--conf", type=float, default=0.05)
    p.add_argument("--iou", type=float, default=0.50)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--device", default=None,
                   help="e.g. 0 for GPU, cpu — default: ultralytics auto")
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args()
    logging.basicConfig(level=getattr(logging, a.log_level),
                        format="%(levelname)s: %(message)s")

    if not a.frames_dir.is_dir():
        LOGGER.error("frames dir not found: %s", a.frames_dir.resolve())
        return 1
    summary = prelabel_dir(
        a.frames_dir, model=a.model, conf=a.conf, iou=a.iou,
        imgsz=a.imgsz, device=a.device,
    )
    if not summary:
        LOGGER.error("no images under %s", a.frames_dir.resolve())
        return 1
    (a.frames_dir / "prelabel_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tot_b = sum(s["boxes"] for s in summary)
    tot_i = sum(s["images"] for s in summary)
    LOGGER.info("Done. %d scenes, %d images, %d pre-labelled boxes",
                len(summary), tot_i, tot_b)
    LOGGER.info("Next (STEP 3): frames_to_cvat.py --frames-dir %s", a.frames_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
