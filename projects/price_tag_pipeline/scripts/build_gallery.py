"""Build a self-contained HTML gallery: crop thumbnail + extracted fields,
sharpest first. The morning deliverable — visual proof that text is read
off the crops. Open data/friends_crops/gallery.html in a browser.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent / "src"))
from price_tag_pipeline import cv_io  # noqa: E402
from price_tag_pipeline.quality import laplacian_sharpness  # noqa: E402
import cv2  # noqa: E402

SHOW = ["product_name", "price_default", "price_card", "price_discount",
        "discount_amount", "barcode", "id_sku", "print_datetime", "code",
        "additional_info", "color", "special_symbols"]

ap = argparse.ArgumentParser()
ap.add_argument("--pred", required=True)
ap.add_argument("--out", default="data/friends_crops/gallery.html")
ap.add_argument("--title", default="Lenta OCR — friends_validated (Qwen3-VL-4B, locked v5)")
args = ap.parse_args()

rows = [json.loads(l) for l in Path(args.pred).read_text(encoding="utf-8").splitlines() if l.strip()]
for r in rows:
    img = cv_io.imread(r["crop_path"])
    r["_sharp"] = laplacian_sharpness(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) if img is not None else 0.0
rows.sort(key=lambda r: -r["_sharp"])

cards = []
for r in rows:
    p = r["pred"]
    cp = Path(r["crop_path"]).resolve().as_uri()
    tr = "".join(
        f"<tr><td class=k>{html.escape(k)}</td><td>{html.escape(str(p.get(k,'')))}</td></tr>"
        for k in SHOW
    )
    cards.append(
        f"<div class=card><img src='{cp}' loading=lazy>"
        f"<div class=meta><div class=cid>{html.escape(r['crop_id'])} "
        f"<span class=sh>sharp {r['_sharp']:.0f}</span></div>"
        f"<table>{tr}</table></div></div>"
    )

doc = f"""<!doctype html><meta charset=utf-8><title>{html.escape(args.title)}</title>
<style>
body{{font:14px/1.4 system-ui;margin:24px;background:#0f1115;color:#e6e6e6}}
h1{{font-size:18px}} .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(440px,1fr));gap:18px}}
.card{{background:#181b22;border:1px solid #2a2f3a;border-radius:10px;overflow:hidden;display:flex;flex-direction:column}}
.card img{{width:100%;max-height:340px;object-fit:contain;background:#000}}
.meta{{padding:10px 12px}} .cid{{font-weight:600;margin-bottom:6px;word-break:break-all}}
.sh{{color:#8b93a7;font-weight:400;font-size:12px}}
table{{border-collapse:collapse;width:100%}} td{{padding:2px 6px;vertical-align:top;border-top:1px solid #232733}}
.k{{color:#8b93a7;width:140px}}
</style>
<h1>{html.escape(args.title)} — {len(rows)} crops, sharpest first</h1>
<div class=grid>{''.join(cards)}</div>"""

out = Path(args.out)
out.write_text(doc, encoding="utf-8")
print(f"gallery: {out.resolve()}  ({len(rows)} crops)")
