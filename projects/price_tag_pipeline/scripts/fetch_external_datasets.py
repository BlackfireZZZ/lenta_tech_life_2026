#!/usr/bin/env python3
"""Fetch external pretraining datasets listed in docs/data/datasets.md.

Safe by default: with no flags it only PRINTS the plan. It downloads a target
only when you pass it explicitly with --download, using your own credentials.
Everything lands under data/external/<name>/ which is git-ignored.

Examples
--------
    # show the plan, download nothing
    python projects/price_tag_pipeline/scripts/fetch_external_datasets.py

    # SKU-110K via the Ultralytics mirror (needs `pip install ultralytics`)
    python .../fetch_external_datasets.py --download sku110k

    # a Roboflow Universe price-tag set (needs ROBOFLOW_API_KEY + `pip install roboflow`)
    ROBOFLOW_API_KEY=xxx python .../fetch_external_datasets.py --download roboflow \
        --rf-workspace cuhk-00cw9 --rf-project price-tag-mpq14 --rf-version 1

    # Kaggle receipts OCR (needs `pip install kaggle` + ~/.kaggle/kaggle.json)
    python .../fetch_external_datasets.py --download kaggle-receipts

    # Grocery Store Dataset (git clone)
    python .../fetch_external_datasets.py --download grocery-store
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PLAN = {
    "sku110k": "SKU-110K dense shelf detection (~11.7k imgs). Detector pretrain. Academic/non-commercial.",
    "roboflow": "Roboflow Universe price-tag set (small, on-domain). Needs ROBOFLOW_API_KEY.",
    "kaggle-receipts": "Kaggle OCR receipts text detection. Needs kaggle.json.",
    "grocery-store": "Klasson Grocery Store Dataset (~5.1k imgs + iconic labels).",
}


def _ext_root() -> Path:
    root = Path(__file__).resolve().parents[3] / "data" / "external"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _run(cmd: list[str], **kw) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, **kw)


def fetch_sku110k(dst: Path) -> None:
    try:
        from ultralytics.utils.downloads import safe_download
    except ModuleNotFoundError:
        sys.exit("Install ultralytics first: pip install ultralytics")
    url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/SKU-110K.zip"
    safe_download(url, dir=dst, unzip=True)
    print(f"SKU-110K -> {dst}")


def fetch_roboflow(dst: Path, workspace: str, project: str, version: int) -> None:
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        sys.exit("Set ROBOFLOW_API_KEY (https://app.roboflow.com -> settings).")
    if not (workspace and project and version):
        sys.exit("Pass --rf-workspace --rf-project --rf-version (see docs/data/datasets.md).")
    try:
        from roboflow import Roboflow
    except ModuleNotFoundError:
        sys.exit("Install roboflow first: pip install roboflow")
    rf = Roboflow(api_key=key)
    ds = rf.workspace(workspace).project(project).version(int(version)).download(
        "yolov8", location=str(dst / f"{workspace}__{project}_v{version}")
    )
    print(f"Roboflow -> {ds.location}")


def fetch_kaggle_receipts(dst: Path) -> None:
    try:
        import kaggle  # noqa: F401
    except (ModuleNotFoundError, OSError):
        sys.exit("Install kaggle + place ~/.kaggle/kaggle.json: pip install kaggle")
    _run(
        ["kaggle", "datasets", "download", "-d",
         "trainingdatapro/ocr-receipts-text-detection",
         "-p", str(dst), "--unzip"]
    )
    print(f"Kaggle receipts -> {dst}")


def fetch_grocery_store(dst: Path) -> None:
    target = dst / "GroceryStoreDataset"
    if target.exists():
        print(f"Already present: {target}")
        return
    _run(["git", "clone", "--depth", "1",
          "https://github.com/marcusklasson/GroceryStoreDataset.git", str(target)])
    print(f"Grocery Store Dataset -> {target}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--download", choices=sorted(PLAN), help="Target to actually download")
    p.add_argument("--rf-workspace", default="")
    p.add_argument("--rf-project", default="")
    p.add_argument("--rf-version", type=int, default=0)
    args = p.parse_args()

    if not args.download:
        print("Plan (nothing downloaded — pass --download <target>):\n")
        for k, v in PLAN.items():
            print(f"  {k:16s} {v}")
        print("\nSee docs/data/datasets.md for licenses and the recommended order of work.")
        return 0

    dst = _ext_root()
    if args.download == "sku110k":
        fetch_sku110k(dst)
    elif args.download == "roboflow":
        fetch_roboflow(dst, args.rf_workspace, args.rf_project, args.rf_version)
    elif args.download == "kaggle-receipts":
        fetch_kaggle_receipts(dst)
    elif args.download == "grocery-store":
        fetch_grocery_store(dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
