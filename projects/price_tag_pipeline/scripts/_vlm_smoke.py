"""Minimal direct VLM smoke — load a local model dir, run it on one real
price-tag crop, print raw output. Used to nail the correct transformers-5.x
load/generate invocation per model before touching the pipeline engine.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent / "src"))
from price_tag_pipeline.recognition.ocr import DEFAULT_VLM_PROMPT  # noqa: E402


def crop_from_label(img_path: Path, label_path: Path, idx: int) -> Image.Image:
    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    lines = [l.split() for l in label_path.read_text().splitlines() if l.strip()]
    _, cx, cy, w, h = (float(v) for v in lines[idx])
    x1 = int((cx - w / 2) * W); y1 = int((cy - h / 2) * H)
    x2 = int((cx + w / 2) * W); y2 = int((cy + h / 2) * H)
    pad_x = int((x2 - x1) * 0.08); pad_y = int((y2 - y1) * 0.08)
    return im.crop((max(0, x1 - pad_x), max(0, y1 - pad_y),
                    min(W, x2 + pad_x), min(H, y2 + pad_y)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--box-idx", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--upscale", type=int, default=3, help="nearest-int upscale of the crop")
    args = ap.parse_args()

    crop = crop_from_label(Path(args.image), Path(args.label), args.box_idx)
    if args.upscale > 1:
        crop = crop.resize((crop.width * args.upscale, crop.height * args.upscale), Image.LANCZOS)
    print(f"crop size (after x{args.upscale}): {crop.size}", flush=True)

    from transformers import AutoProcessor
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)

    model = None
    last_err = None
    for cls_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModelForCausalLM"):
        try:
            import transformers
            cls = getattr(transformers, cls_name)
            model = cls.from_pretrained(
                args.model_dir, trust_remote_code=True,
                torch_dtype=torch.bfloat16, device_map="cuda",
            )
            print(f"loaded via {cls_name} in {time.time()-t0:.1f}s", flush=True)
            break
        except Exception as e:  # noqa: BLE001
            last_err = f"{cls_name}: {type(e).__name__}: {e}"
            print("  x " + last_err[:300], flush=True)
    if model is None:
        print("ALL LOADERS FAILED:", last_err)
        return 1
    model.eval()

    messages = [{"role": "user", "content": [
        {"type": "image", "image": crop},
        {"type": "text", "text": DEFAULT_VLM_PROMPT},
    ]}]
    inputs = proc.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    t1 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    gen = out[:, inputs["input_ids"].shape[1]:]
    text = proc.batch_decode(gen, skip_special_tokens=True)[0]
    print(f"--- generate {time.time()-t1:.1f}s | VRAM {torch.cuda.max_memory_allocated()/1e9:.1f}GB ---")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
