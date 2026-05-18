"""Score run_on_crops predictions against the curated friends reference.

Metric fidelity: reuses eval_hack_csv field logic (the only faithful
hackathon scorer) for numeric/"нет"/string fields; product_name &
additional_info use CER<=tau (organizers said the name need not be
char-perfect — task §5.3 note). Only fields PRESENT in the reference are
scored (a missing ref field = unknown, skipped). 1:1 by crop_id.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

THIS = Path(__file__).resolve().parent
for p in (str(THIS.parent / "src"), str(THIS)):
    if p not in sys.path:
        sys.path.insert(0, p)

import eval_hack_csv as ehc  # noqa: E402
from price_tag_pipeline.metrics.ocr_metrics import cer  # noqa: E402

CER_FIELDS = {"product_name", "additional_info"}


def _match(field: str, pred: str, ref: str, numeric_tol: float, name_tau: float) -> bool:
    rn = ehc._norm_text(ref)
    if not rn:
        return True  # unknown ref -> skip at caller
    if field in CER_FIELDS and rn != "нет":
        pn = ehc._norm_text(pred)
        if not pn:
            return False
        return cer(pn, rn) <= name_tau
    return ehc._field_match(field, pred, ref, numeric_tol=numeric_tol)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="run_on_crops JSONL")
    ap.add_argument("--ref", required=True, help="curated reference JSONL")
    ap.add_argument("--numeric-tol", type=float, default=0.01)
    ap.add_argument("--name-tau", type=float, default=0.20)
    ap.add_argument("--tag-pass", type=float, default=0.80)
    ap.add_argument("--show-misses", action="store_true")
    args = ap.parse_args()

    pred = {}
    for l in Path(args.pred).read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            pred[r["crop_id"]] = r["pred"]
    ref = {}
    for l in Path(args.ref).read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            ref[r["crop_id"]] = r

    fields = ehc.DEFAULT_GRADED_FIELDS
    fc, fco = Counter(), Counter()
    per_tag = []
    scored_ids = []
    for cid, refrow in ref.items():
        if cid not in pred:
            continue
        prow = pred[cid]
        considered = correct = 0
        misses = []
        for f in fields:
            if f not in refrow:
                continue
            rv = str(refrow.get(f, ""))
            if not ehc._norm_text(rv):
                continue
            considered += 1
            fco[f] += 1
            ok = _match(f, str(prow.get(f, "")), rv, args.numeric_tol, args.name_tau)
            if ok:
                correct += 1
                fc[f] += 1
            elif args.show_misses:
                misses.append(f"{f}: ref={rv!r} pred={prow.get(f)!r}")
        if considered:
            per_tag.append(correct / considered)
            scored_ids.append(cid)
            if args.show_misses and misses:
                print(f"[{cid}] {correct}/{considered}")
                for m in misses:
                    print("   ", m)

    n = len(per_tag)
    passed = sum(1 for s in per_tag if s >= args.tag_pass)
    print("\n" + "=" * 56)
    print(f"friends scoreboard | scored {n} crops")
    print(f"mean per-tag accuracy:        {sum(per_tag)/n if n else 0:.4f}")
    print(f"share >= {args.tag_pass:.2f}:           {passed}/{n} = "
          f"{passed/n if n else 0:.4f}   <<< headline")
    print("-" * 56)
    for f in sorted(fields, key=lambda f: (fc[f]/fco[f]) if fco[f] else 9):
        if fco[f]:
            print(f"  {f:24s} {fc[f]/fco[f]:.3f}  [{fc[f]}/{fco[f]}]")
    print("=" * 56)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
