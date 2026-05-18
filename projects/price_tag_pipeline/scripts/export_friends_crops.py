"""Export friends_validated good-box price-tag crops (padded + LANCZOS
upscaled) + a manifest. These good boxes are the OCR scoreboard substrate
(organizer gt_e2e boxes are garbage — see OCR_CAMPAIGN_LOG.md).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent / "src"))
from price_tag_pipeline import cv_io  # noqa: E402
import cv2  # noqa: E402

ROOT = Path("E:/Hackatons/lenta_tech_life_2026/.claude/worktrees/worktree-ocr")
FR = ROOT / "raw_photos" / "frames" / "friends_validated"
LB = ROOT / "raw_photos" / "annotations" / "labels" / "friends_validated"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "friends_crops"))
    ap.add_argument("--pad", type=float, default=0.10)
    ap.add_argument("--target", type=int, default=1024, help="upscale so max side >= this")
    ap.add_argument("--limit-images", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    imgs = sorted(FR.glob("*.jpg"))
    if args.limit_images:
        imgs = imgs[: args.limit_images]

    manifest = []
    n = 0
    for img_path in imgs:
        lbl = LB / (img_path.stem + ".txt")
        if not lbl.exists():
            continue
        frame = cv_io.imread(str(img_path))
        if frame is None:
            continue
        H, W = frame.shape[:2]
        boxes = [ln.split() for ln in lbl.read_text().splitlines() if ln.strip()]
        for i, b in enumerate(boxes):
            cx, cy, bw, bh = (float(b[1]), float(b[2]), float(b[3]), float(b[4]))
            x1 = (cx - bw / 2) * W
            y1 = (cy - bh / 2) * H
            x2 = (cx + bw / 2) * W
            y2 = (cy + bh / 2) * H
            px = (x2 - x1) * args.pad
            py = (y2 - y1) * args.pad
            X1 = max(0, int(x1 - px)); Y1 = max(0, int(y1 - py))
            X2 = min(W, int(x2 + px)); Y2 = min(H, int(y2 + py))
            if X2 - X1 < 8 or Y2 - Y1 < 8:
                continue
            crop = frame[Y1:Y2, X1:X2].copy()
            ch, cw = crop.shape[:2]
            scale = max(1.0, args.target / max(ch, cw))
            if scale > 1.0:
                crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)),
                                  interpolation=cv2.INTER_LANCZOS4)
            cid = f"{img_path.stem}__{i}"
            cpath = out / f"{cid}.jpg"
            cv_io.imwrite(str(cpath), crop)
            manifest.append({
                "crop_id": cid, "src_image": img_path.name, "box_idx": i,
                "crop_path": str(cpath), "orig_px": [X2 - X1, Y2 - Y1],
                "out_px": [crop.shape[1], crop.shape[0]],
            })
            n += 1
    (out / "manifest.jsonl").write_text(
        "\n".join(json.dumps(m, ensure_ascii=False) for m in manifest) + "\n",
        encoding="utf-8")
    print(f"exported {n} crops from {len(imgs)} images -> {out}")
    print(f"manifest: {out / 'manifest.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
