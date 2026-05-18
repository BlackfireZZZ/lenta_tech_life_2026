"""Assemble loadable local HF model dirs from the user's flat ./weights dump
+ the small config/tokenizer files fetched into the HF cache.

Identity was verified by matching each repo's model.safetensors.index.json
total_size (and exact shard filenames) to the user's blobs:

  weights/model.safetensors                 (2528MB single) -> GLM-OCR
  weights/model-0000{1,2}-of-00002          (8466MB 2-shard) -> Qwen3-VL-4B
  weights/model-0000{1..4}-of-00004         (1903MB 4-shard) -> HunyuanOCR

Big safetensors are MOVED (instant, same filesystem, no disk duplication) into
weights/<slug>/; small config files are copied from the HF snapshot.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
from pathlib import Path

WT = Path(__file__).resolve().parents[3] / "weights"
CACHE = Path(os.path.expanduser("~/.cache/huggingface/hub"))


def snap(repo_dir: str) -> Path:
    s = sorted((CACHE / repo_dir / "snapshots").glob("*"))
    if not s:
        raise SystemExit(f"no snapshot for {repo_dir}")
    return s[-1]


PLAN = {
    "glm-ocr": {
        "snap": "models--zai-org--GLM-OCR",
        "blobs": {"model.safetensors": "model.safetensors"},
    },
    "qwen3-vl-4b": {
        "snap": "models--Qwen--Qwen3-VL-4B-Instruct",
        "blobs": {
            "model-00001-of-00002.safetensors": "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors": "model-00002-of-00002.safetensors",
        },
    },
    "hunyuan-ocr": {
        "snap": "models--tencent--HunyuanOCR",
        "blobs": {
            "model-00001-of-00004.safetensors": "model-00001-of-00004.safetensors",
            "model-00002-of-00004.safetensors": "model-00002-of-00004.safetensors",
            "model-00003-of-00004.safetensors": "model-00003-of-00004.safetensors",
            "model-00004-of-00004.safetensors": "model-00004-of-00004.safetensors",
        },
    },
}

manifest = {}
for slug, spec in PLAN.items():
    dst = WT / slug
    dst.mkdir(parents=True, exist_ok=True)
    sp = snap(spec["snap"])
    # copy every small (non-weight) file from the snapshot
    copied = []
    for f in os.listdir(sp):
        src = sp / f
        if not src.is_file():
            continue
        shutil.copy2(src, dst / f)
        copied.append(f)
    # move the big blobs in (instant rename within E:)
    moved = []
    for src_name, dst_name in spec["blobs"].items():
        src = WT / src_name
        target = dst / dst_name
        if target.exists():
            moved.append(f"{dst_name} (already present)")
            continue
        if not src.exists():
            moved.append(f"!! MISSING {src_name}")
            continue
        try:
            os.replace(src, target)  # same filesystem -> instant move
        except OSError:
            shutil.move(str(src), str(target))
        moved.append(f"{src_name} -> {dst_name}")
    sz = sum(p.stat().st_size for p in dst.glob("*.safetensors")) / 1048576
    manifest[slug] = {"dir": str(dst), "configs": sorted(copied),
                      "weights": moved, "weights_MB": round(sz)}

print(json.dumps(manifest, ensure_ascii=False, indent=2))
print("\nLeftover in weights/ root:",
      sorted(p.name for p in WT.glob("*.safetensors")))
