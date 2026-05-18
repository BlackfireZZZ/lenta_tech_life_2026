#!/usr/bin/env python3
"""Apply Lenta-catalog reconciliation to a hackathon-format CSV.

Reads a predictions/GT CSV (the 29-column hackathon schema), reconciles each
row's ``barcode`` + ``product_name`` against ``real_data/db_hack.csv``, writes
a corrected CSV (all other columns preserved), and prints a change report.

Examples
--------
    # quick one-off probe (no CSV needed)
    python catalog_reconcile.py --probe "4 607018 308355" "Коктейль высокб"

    # correct a predictions CSV
    python catalog_reconcile.py --in preds.csv --out preds.reconciled.csv

    # measure recovery against the labeled GT
    python catalog_reconcile.py --in real_data/dataset/49_5/49_5.csv \
        --out /tmp/49_5.recon.csv --report /tmp/49_5.report.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

# Make the package importable when run as a plain script.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from price_tag_pipeline.catalog import CatalogIndex, CatalogReconciler  # noqa: E402


def _default_catalog() -> Path:
    """Find ``real_data/db_hack.csv`` in this tree or the main checkout.

    ``real_data/`` is gitignored and lives only in the primary checkout, not in
    git worktrees — so walk up from both the script and the CWD.
    """
    rel = Path("real_data") / "db_hack.csv"
    seeds = [Path(__file__).resolve(), Path.cwd().resolve()]
    for seed in seeds:
        for base in [seed, *seed.parents]:
            cand = base / rel
            if cand.is_file():
                return cand
    return rel  # let the error surface clearly downstream


def _probe(rec: CatalogReconciler, barcode: str, name: str) -> int:
    r = rec.reconcile(barcode, name)
    print(json.dumps(
        {
            "in": {"barcode": barcode, "product_name": name},
            "out": {"barcode": r.barcode, "product_name": r.product_name},
            "barcode_source": r.barcode_source,
            "name_source": r.name_source,
            "name_score": r.name_score,
            "checksum_ok": r.checksum_ok,
            "barcode_in_catalog": r.barcode_in_catalog,
            "conflict": r.conflict,
            "notes": list(r.notes),
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalog", type=Path, default=None,
                    help="db_hack.csv path (default: autodetected)")
    ap.add_argument("--in", dest="inp", type=Path,
                    help="input hackathon CSV")
    ap.add_argument("--out", type=Path, help="output corrected CSV")
    ap.add_argument("--report", type=Path,
                    help="optional JSON change-report path")
    ap.add_argument("--diff", type=Path,
                    help="optional JSONL of every changed row")
    ap.add_argument("--limit", type=int, default=0,
                    help="process only the first N rows (0 = all)")
    ap.add_argument("--no-cache", action="store_true",
                    help="ignore/skip the on-disk index cache")
    ap.add_argument(
        "--name-policy", choices=["fill", "fix"], default="fill",
        help="fill: only fill a missing name (safe default). "
        "fix: also rewrite a present name to the canonical catalog spelling.",
    )
    ap.add_argument("--recover-cutoff", type=float, default=92.0)
    ap.add_argument("--name-fix-cutoff", type=float, default=96.0)
    ap.add_argument("--conflict-cutoff", type=float, default=55.0)
    ap.add_argument("--probe", nargs=2, metavar=("BARCODE", "NAME"),
                    help="reconcile a single pair and exit")
    args = ap.parse_args(argv)

    catalog_path = args.catalog or _default_catalog()
    if not Path(catalog_path).is_file():
        ap.error(f"catalog not found: {catalog_path}")

    t0 = time.time()
    index = CatalogIndex.load(catalog_path, cache=not args.no_cache)
    print(
        f"catalog: {len(index):,} rows "
        f"({index.barcode_collisions:,} dup barcodes) "
        f"loaded in {time.time() - t0:.1f}s",
        file=sys.stderr,
    )
    rec = CatalogReconciler(
        index,
        name_policy=args.name_policy,
        name_recover_cutoff=args.recover_cutoff,
        name_only_fix_cutoff=args.name_fix_cutoff,
        conflict_cutoff=args.conflict_cutoff,
    )

    if args.probe:
        return _probe(rec, args.probe[0], args.probe[1])

    if not args.inp or not args.out:
        ap.error("--in and --out are required (unless --probe)")
    if not args.inp.is_file():
        ap.error(f"input CSV not found: {args.inp}")

    with args.inp.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    if "barcode" not in fieldnames or "product_name" not in fieldnames:
        ap.error(
            "input CSV must have 'barcode' and 'product_name' columns "
            f"(got: {fieldnames})"
        )
    if args.limit:
        rows = rows[: args.limit]

    stats: Counter[str] = Counter()
    diff_fh = args.diff.open("w", encoding="utf-8") if args.diff else None
    try:
        for row in rows:
            stats["rows"] += 1
            r = rec.reconcile(row.get("barcode"), row.get("product_name"))
            if r.barcode_in_catalog:
                stats["barcode_in_catalog"] += 1
            if r.conflict:
                stats["conflicts"] += 1
            if r.barcode_source != "kept":
                stats[f"barcode:{r.barcode_source}"] += 1
            if r.name_source != "kept":
                stats[f"name:{r.name_source}"] += 1
            if r.changed and diff_fh is not None:
                diff_fh.write(json.dumps({
                    "filename": row.get("filename"),
                    "before": {"barcode": row.get("barcode"),
                               "product_name": row.get("product_name")},
                    "after": {"barcode": r.barcode,
                              "product_name": r.product_name},
                    "barcode_source": r.barcode_source,
                    "name_source": r.name_source,
                    "name_score": r.name_score,
                    "conflict": r.conflict,
                    "notes": list(r.notes),
                }, ensure_ascii=False) + "\n")
            row["barcode"] = r.barcode if r.barcode is not None else ""
            row["product_name"] = (
                r.product_name if r.product_name is not None else ""
            )
    finally:
        if diff_fh is not None:
            diff_fh.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "input": str(args.inp),
        "output": str(args.out),
        "catalog": str(catalog_path),
        "catalog_rows": len(index),
        "stats": dict(sorted(stats.items())),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.report:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
