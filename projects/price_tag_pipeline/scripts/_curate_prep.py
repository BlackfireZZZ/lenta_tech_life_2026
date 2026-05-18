"""Triage helper for building the curated friends reference: rank the
stratified crops by Laplacian sharpness and join the Qwen draft so the
sharpest, most-legible tags are reviewed/corrected first.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent / "src"))
from price_tag_pipeline import cv_io  # noqa: E402
from price_tag_pipeline.quality import laplacian_sharpness  # noqa: E402
import cv2  # noqa: E402

ROOT = Path("E:/Hackatons/lenta_tech_life_2026/.claude/worktrees/worktree-ocr")
draft = {}
dp = ROOT / "outputs" / "qwen_draft_1perimg.jsonl"
if dp.exists():
    for l in dp.read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            draft[r["crop_id"]] = r

man = {json.loads(l)["crop_id"]: json.loads(l)
       for l in (ROOT / "data/friends_crops/manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}

rows = []
for cid, d in draft.items():
    img = cv_io.imread(d["crop_path"])
    if img is None:
        continue
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = laplacian_sharpness(g)
    p = d["pred"]
    rows.append((round(s, 1), cid, d["crop_path"],
                 {k: p.get(k) for k in ("product_name", "price_default", "price_card",
                  "price_discount", "barcode", "discount_amount", "weight_value",
                  "weight_unit", "color", "special_symbols") if p.get(k) and p.get(k) != "нет"}))
rows.sort(reverse=True)
out = ROOT / "outputs" / "curate_ranked.jsonl"
with out.open("w", encoding="utf-8") as f:
    for s, cid, path, pred in rows:
        f.write(json.dumps({"sharpness": s, "crop_id": cid, "crop_path": path,
                            "qwen": pred}, ensure_ascii=False) + "\n")
print(f"ranked {len(rows)} crops by sharpness -> {out}")
for s, cid, _, pred in rows[:20]:
    print(f"  {s:7.1f}  {cid}  {pred}")
