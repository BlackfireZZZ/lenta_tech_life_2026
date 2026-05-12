#!/usr/bin/env python3
"""Build video-level GroupKFold split manifests.

If `data/raw/metadata.csv` exists with columns `video_id,store_id,aisle_id,...`,
splits are stratified by `(store_id, aisle_id)`. Otherwise a plain shuffled KFold.

Usage:
    python projects/price_tag_pipeline/scripts/make_splits.py \\
        --processed data/processed \\
        --metadata data/raw/metadata.csv \\
        --out data/splits \\
        --n_splits 5
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

from price_tag_pipeline.data.splits import (  # noqa: E402
    build_video_level_folds,
    load_metadata,
    write_folds,
)
from price_tag_pipeline.data.loaders import write_dataset_yaml  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--processed", default="data/processed")
    p.add_argument("--metadata", default="data/raw/metadata.csv")
    p.add_argument("--out", default="data/splits")
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--emit-dataset-yaml-fold", type=int, default=0,
                   help="Re-emit dataset.yaml with this fold's train/val lists.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    processed = Path(args.processed).resolve()
    frames_root = processed / "frames"
    if not frames_root.exists():
        logging.error("No frames found under %s. Run prepare_data.py first.", frames_root)
        return 1

    video_ids = sorted(d.name for d in frames_root.iterdir() if d.is_dir())
    if not video_ids:
        logging.error("No video subfolders under %s.", frames_root)
        return 1

    metadata_path = Path(args.metadata)
    metadata = load_metadata(metadata_path)
    if not metadata:
        logging.info("No metadata at %s; building unstratified KFold.", metadata_path)
    else:
        logging.info("Loaded metadata for %d videos.", len(metadata))

    folds = build_video_level_folds(video_ids, n_splits=args.n_splits, metadata=metadata, seed=args.seed)
    out_dir = Path(args.out).resolve()
    paths = write_folds(folds, out_dir)
    for p_ in paths:
        logging.info("Wrote %s", p_)

    # Conveniently update dataset.yaml for the requested fold.
    fold_id = args.emit_dataset_yaml_fold
    if 0 <= fold_id < len(folds):
        # Read classes from existing dataset.yaml.
        ds_yaml = processed / "dataset.yaml"
        classes: list[str] = []
        if ds_yaml.exists():
            for line in ds_yaml.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith(("0:", "1:", "2:", "3:", "4:", "5:", "6:", "7:", "8:", "9:")):
                    classes.append(line.split(":", 1)[1].strip())
        if not classes:
            classes = ["price_tag"]
        write_dataset_yaml(processed, classes, fold=folds[fold_id], fold_id=fold_id)
        logging.info("Updated %s with fold %d", ds_yaml, fold_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
