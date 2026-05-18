"""Fetch ONLY the small repo files (configs/tokenizer/processor/index/modeling
.py) for the VLM backends — never the multi-GB weights (the user already
downloaded those into ./weights). Lets us identify each safetensors blob by
its repo's model.safetensors.index.json total_size and assemble loadable dirs.
"""
from __future__ import annotations

import sys
from huggingface_hub import snapshot_download

REPOS = [
    "zai-org/GLM-OCR",
    "Qwen/Qwen3-VL-4B-Instruct",
    "Tencent-Hunyuan/HunyuanOCR",
]
IGNORE = ["*.safetensors", "*.bin", "*.pth", "*.pt", "*.gguf",
          "*.h5", "*.msgpack", "*.onnx", "*.ot"]

for repo in REPOS:
    try:
        path = snapshot_download(repo, ignore_patterns=IGNORE)
        print(f"OK   {repo} -> {path}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL {repo}: {type(exc).__name__}: {exc}", flush=True)
print("DONE", flush=True)
sys.exit(0)
