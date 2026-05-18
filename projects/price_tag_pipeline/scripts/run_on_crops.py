"""Run the recognition chain over a set of crop images → JSONL of
{crop_id, raw_text, pred_row(29-col)}. Reused for the friends scoreboard
scoring and the morning gallery. No GT needed here.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent / "src"))

from price_tag_pipeline import cv_io
from price_tag_pipeline.config import RecognitionConfig, load_config
from price_tag_pipeline.recognition.chain import build_recognition_chain
from price_tag_pipeline.submission import hack_row_from_tag_dict
from price_tag_pipeline.types import HACK_EXTRA_FIELDS, ParsedTag


def parsed_to_row(parsed: ParsedTag, filename: str) -> dict:
    d = {
        "bbox": [0, 0, 0, 0], "timestamp_s": 0.0,
        "regular_price": parsed.regular_price,
        "loyalty_price": parsed.loyalty_price,
        "product_name": parsed.product_name,
        "weight_value": parsed.weight_value,
        "weight_unit": parsed.weight_unit.value if parsed.weight_unit else None,
        "price_per_unit_value": parsed.price_per_unit_value,
        "price_per_unit_unit": parsed.price_per_unit_unit,
        "promo_flag": bool(parsed.promo_flag), "currency": parsed.currency,
    }
    for k in HACK_EXTRA_FIELDS:
        d[k] = parsed.extra_fields.get(k)
    return hack_row_from_tag_dict(d, filename)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(THIS.parents[2] / "data" / "friends_crops" / "manifest.jsonl"))
    ap.add_argument("--backend", default="glm_ocr")
    ap.add_argument("--vlm-model", required=True)
    ap.add_argument("--config", default=str(THIS.parent / "configs" / "balanced.yaml"))
    ap.add_argument("--ids", default="", help="comma crop_ids; empty=all")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=640)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg = dataclasses.replace(
        cfg,
        ocr=dataclasses.replace(cfg.ocr, backend=args.backend, vlm_model=args.vlm_model,
                                vlm_max_new_tokens=args.max_new_tokens),
        recognition=RecognitionConfig(enable_qr=False, enable_barcode=False, enable_ocr=True),
    )
    chain = build_recognition_chain(cfg)

    rows = [json.loads(l) for l in Path(args.manifest).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.ids:
        want = set(args.ids.split(","))
        rows = [r for r in rows if r["crop_id"] in want]
    if args.limit:
        rows = rows[: args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fh = out.open("w", encoding="utf-8")
    t0 = time.time()
    for i, m in enumerate(rows):
        img = cv_io.imread(m["crop_path"])
        if img is None:
            continue
        results = chain.decode(img)
        best = max(results, key=lambda r: (r.found, r.confidence)) if results else None
        parsed = best.parsed if best else ParsedTag()
        raw = best.text if best else ""
        rec = {"crop_id": m["crop_id"], "crop_path": m["crop_path"],
               "raw_text": raw[:1200], "pred": parsed_to_row(parsed, m["crop_id"])}
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        if (i + 1) % 10 == 0:
            print(f"{i+1}/{len(rows)}  {(time.time()-t0)/(i+1):.1f}s/crop", flush=True)
    fh.close()
    print(f"done {len(rows)} crops in {time.time()-t0:.0f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
