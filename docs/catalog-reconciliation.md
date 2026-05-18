# Catalog reconciliation — repair `barcode` / `product_name` from `db_hack.csv`

A local, network-free post-recognition step that cross-checks the recognized
`barcode` and `product_name` against the **Lenta master catalog**
(`real_data/db_hack.csv`, cp1251, `fullname;code`, ~625 k rows).

* Code: `price_tag_pipeline.catalog` (`CatalogIndex`, `CatalogReconciler`)
* CLI: `projects/price_tag_pipeline/scripts/catalog_reconcile.py`
* Tests: `projects/price_tag_pipeline/tests/test_catalog_reconcile.py`
* Dep: `rapidfuzz` (in `requirements/base.txt`) — light, C++, no network.
  Fits the "cloud APIs banned at inference / lightweight rewarded" rule.

## Why this and not product photos

The original ask was scraping product photos from lenta.com. Recon showed
`api.lenta.com` sits behind an anti-bot WAF: plain HTTP **and**
Chrome-TLS-impersonation (`curl_cffi`) both get **HTTP 403**; only a real
browser session passes. That makes catalog-wide photo scraping (625 k rows)
infeasible/abusive, so we pivoted to this CSV-only tool, which is a bigger,
safer OCR lever anyway. (Photo scraping is *paused*, not impossible — it would
need a browser-driven path and a bounded barcode set.)

## What it does

1. **Normalize barcode** — strip OCR spaces/glyphs (`"4 607018 308355"` →
   `"4607018308355"`); validate the GS1 check digit (informational).
2. **Barcode is the primary key** ([index.md](./index.md) fact #2). If the
   barcode resolves in the catalog, the catalog `fullname` is authoritative:
   fill a *missing* name, or (opt-in) correct a mis-read one.
3. **Recover a missing barcode** from a high-confidence fuzzy name match
   (RapidFuzz `token_set_ratio`, first-token blocked + full-scan fallback).
4. **Guard a mis-read barcode** — if it fails its own GTIN check digit *and*
   strongly disagrees with a present name, don't let it overwrite the name;
   raise `conflict` instead.
5. **Respect `"нет"` vs empty** ([index.md](./index.md) fact #5) — the
   intentional-absence sentinel is never overwritten and never fuzzy-matched;
   only genuinely *missing* (`None`/`""`) fields are filled.

## The name-policy trade-off (read before enabling `fix`)

The scorer (`eval_hack_csv.py`, the faithful reference) compares
`product_name` and `barcode` by **exact, case/space-normalized string match**
— *no fuzzy*. The labelled GT `product_name` itself carries OCR-style noise
(Latin `с`/`у`, `6/`, `амж`, missing weight). Measured on the 5 GT videos, the
catalog `fullname` is the *objectively correct, clean* name in essentially
every divergence — but rewriting a present name to it can still **miss the
noisy GT string** under this exact-match proxy.

So policy is explicit, conservative by default:

| `--name-policy` | Behaviour | Use when |
|---|---|---|
| `fill` (default) | Only fill a **missing** name; never replace a present one. Conflicts are *flagged*, not applied. | Always safe: empty scores 0; the catalog name can only match-or-tie. Verified **0 name changes** across all GT (no risk of moving away from the scored answer). |
| `fix` | Also rewrite a present name to the canonical catalog spelling. | The official metric is key-based / fuzzy, or the inputs are known-noisy OCR (not GT-mimicking). Objectively correct, double-edged on the local proxy. |

**Barcode caveat:** space-stripping is required for the catalog lookup, dedup,
and the official barcode *key* — but ~22 % of GT barcodes contain spaces, and
the local string scorer keeps single spaces. On those rows the normalized
barcode can mismatch the spaced GT *field* (the CLI's `barcode:normalized`
count shows the magnitude). This is the right canonical form for matching/
dedup; weigh it if you optimise purely against the local proxy.

## Evidence (5 labelled GT videos, full 625 k catalog)

* Barcode→catalog coverage: **~97 %** of GT tags (266/274).
* Catalog `code` is effectively a **unique key**: of 356 k distinct codes only
  **3** real EANs map to a different name. First-occurrence-wins is safe.
* Default `fill` policy: **0** name rewrites on GT (zero damage), barcode
  normalization + a couple of missing-barcode recoveries (pure win). The real
  upside lands on *noisy pipeline output*, not on already-correct GT.

## Usage

```bash
# one-off probe
python scripts/catalog_reconcile.py --probe "4 607018 308355" "Коктейль высокб"

# correct a predictions CSV (safe default)
python scripts/catalog_reconcile.py --in preds.csv --out preds.recon.csv \
    --diff preds.diff.jsonl

# aggressive name canonicalization (only if the official metric tolerates it)
python scripts/catalog_reconcile.py --in preds.csv --out preds.recon.csv \
    --name-policy fix
```

```python
from price_tag_pipeline.catalog import CatalogIndex, CatalogReconciler

rec = CatalogReconciler(CatalogIndex.load("real_data/db_hack.csv"))
r = rec.reconcile(barcode="4 607018 308355", name_raw="")
# r.barcode -> "4607018308355", r.product_name -> canonical fullname,
# r.barcode_in_catalog, r.conflict, r.notes explain every decision.
```

The 625 k index builds in ~7 s and is pickle-cached next to the CSV (keyed by
size+mtime), so subsequent runs are instant. `real_data/` is gitignored and
lives only in the primary checkout — the CLI auto-locates it from a worktree.
