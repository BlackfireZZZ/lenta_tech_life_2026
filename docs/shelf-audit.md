# Shelf Audit — killer feature plan (out-of-stock + missing-price-tag + product cards)

Status: **plan, not yet implemented.** Worktree `worktree-shelf-analytics`
(forked from `main` @ `e2fd81b9`). This is the actionable build plan for the
hackathon "killer feature". The existing [`shelf-analytics.md`](./shelf-analytics.md)
is the friend's broader research design; this doc is the concrete, scoped
subset we will actually ship (his options 1 + 3).

## 1. What we ship and what the jury sees

A second, fully-automatic analysis path that runs **beside** the graded
29-column price-tag CSV (never into it). For an uploaded robot video it
produces:

1. **Alerts feed** — two alert types, with evidence crops + timestamp + aisle:
   - `MISSING_PRICE_TAG` — a product is on the shelf with **no price tag**
     (a real ЗоЗПП / consumer-law violation in RU retail; high jury value).
   - `OUT_OF_STOCK` — a price tag with **no product** above it (empty shelf =
     lost sales; the single most valuable retail-ops signal).
2. **Product cards** — one card per distinct product seen: best photo crop,
   facing count, and (when a tag was matched) price + name + barcode, with the
   barcode/name canonicalised against the local master catalog.

Demo headline number: *"N пустых полок, M товаров без ценника на видео;
≈X₽ потенциальных потерь"*. This turns a price-reader robot into a
shelf-audit robot — the business reason the robot exists.

> **Hidden rule (project memory `jury-facing-no-manual-annotation`):** no
> jury-facing artifact (UI text, demo script, this feature's pitch) may mention
> manual annotation / CVAT / hand-labelled GT. This whole path is automatic, so
> stay silent on labelling and never imply it.

## 2. Hard separation guarantee

- New code lives only in `price_tag_pipeline/shelf_analytics/` (extend) + one
  new runner script + one optional frontend page.
- Output goes to `outputs/shelf_audit/<video>/` (`audit.json`, `alerts.json`,
  `cards/*.jpg`). **Never** touch `submission.py` or the 29-column CSV path.
- The scored pipeline keeps emitting `submission/hack_submission.csv` unchanged.
  This path is purely additive and can be disabled with one flag.

## 3. Architecture

```text
robot video ─┬─► price-tag pipeline (EXISTING, unchanged)
             │       PriceTagPipeline(cfg).run(video) ─► list[FinalTag]
             │
             └─► product-facing detector (REUSE build_detector + stream_video,
                     second DetectorConfig → product_facing .pt, rotate=ccw,
                     BoT-SORT) ─► tracked product Detection[] (orig-frame coords)
                                   │
            FinalTag[]  +  product Detection[]
                                   │
                                   ▼
             track-level temporal associator (NEW, reuses geometry + matcher
             scoring) ─► relations + UNMATCHED tags + UNMATCHED products
                                   │
                ┌──────────────────┼───────────────────┐
                ▼                  ▼                    ▼
          OUT_OF_STOCK      MISSING_PRICE_TAG     product card builder
          (tag, no prod)    (prod, no tag)        (best crop + tag info +
                                                   catalog reconcile + dedup)
                                   │
                                   ▼
             outputs/shelf_audit/<video>/{audit.json, alerts.json, cards/*}
                                   │
                                   ▼
                    optional UI page (mirrors PipelinePage)
```

Why this is low-risk: both detectors share the **same proven streaming/rotation/
tracking infra** (`detector.py`), so product boxes land in the *same
original-frame coordinate space and timeline* as `FinalTag`. Association is
then pure geometry over a shared coordinate system — no re-projection bugs.

## 4. Reuse decision (friend's code)

Audited every file. Verdict per the user's rule "reuse only if genuinely good":

| File | Quality | Decision |
|---|---|---|
| `detector.py` `build_detector`/`stream_video`/`unrotate_box_xyxy` | Excellent, battle-tested | **Reuse verbatim** — second `DetectorConfig` only |
| `shelf_analytics/geometry.py` | Correct, minimal | **Reuse verbatim** |
| `shelf_analytics/schema.py` | Clean frozen dataclasses | **Reuse + extend** (add alert/card types + a status enum) |
| `shelf_analytics/matcher.py` `_score`/`_text_similarity` | Sound scoring weights | **Reuse the scoring helpers**; replace the per-frame tag→group loop |
| `shelf_analytics/product_detector.py` `product_facing_from_detection` | Trivial adapter | **Reuse** |
| `catalog/` (index/normalize/reconcile) | Shipped + tested | **Reuse** for card barcode/name canonicalisation |
| `progress.py` | Stable event seam | **Reuse** for the runner's progress bar |
| `shelf_analytics/grouping.py` | Per-*image* facing cluster, arbitrary 0.86 thresh | **Mostly superseded** for our run-scale, track-based cards; keep (tested, separate) but not on the critical path |
| `shelf_analytics/pipeline.py` `build_shelf_state*` | Single-image API | **Keep**, but add a new run-scale `build_shelf_audit()` rather than force-fit |

Net: ~65% reuse, new code is the run-scale associator, card/catalog builder,
audit schema, runner, optional UI.

## 5. Integration seams (concrete)

- **Product detector config.** New runtime profile/`DetectorConfig`:
  `backend="yolo"`, `model_path=<abs path to>
  data/checkpoints/product_detector/product_facing_retail_pretrain_yolo11s_best.pt`,
  `frame_rotation="ccw"` (memory `detector-needs-upright-frames`: robot cam is
  90° CW; the model was trained on upright SKU-110K so it *must* see upright
  frames), `image_size=1280` (trained at 1280; dense small objects),
  `tracker_yaml=configs/trackers/bytetrack.yaml`, tuned `conf` (P0 calibrates).
  The checkpoint is `.gitignore`d and absent from the worktree → default to the
  main working-tree absolute path, overridable by env/flag.
- **Price-tag side.** Call the existing `PriceTagPipeline(cfg).run(video)` →
  `list[FinalTag]`; adapt via existing `price_tag_from_final_tag()`.
- **Coordinate/time space (P0-confirmed, critical).** `stream_video`
  un-projects rotated boxes back to **original (sideways) frame coords** —
  `FinalTag.bbox_xyxy` and our product boxes are *both* in that original space,
  so they share a coordinate system. **But the matcher's "tag sits below
  product" geometry is only valid in CCW-upright space.** The associator must
  forward-rotate both boxes into upright space first, using
  `_to_upright(box, orig_w)` — the analytic inverse of
  `detector.unrotate_box_xyxy(..., 'ccw')` (`x_up = oy`, `y_up = orig_w - ox`).
  P0 evidence (video `25_12-20`): tag-above-product ratio = **0.60 in original
  space vs 1.00 in upright space**. Time is already shared
  (`.timestamp_s`/`.source_frames`/`frame_idx`).

## 6. Data contracts (new, in `schema.py`)

```jsonc
// outputs/shelf_audit/<video>/audit.json
{
  "video_id": "video_03",
  "fps": 25.0,
  "summary": { "n_out_of_stock": 4, "n_missing_price_tag": 7,
               "n_products": 38, "n_price_tags": 41,
               "est_lost_revenue_rub": 5230.0 },
  "relations": [ { "price_tag_track_id": 12, "product_track_ids": [88],
                   "score": 0.81, "status": "ok",
                   "reasons": ["horizontal_overlap","tag_below_group"] } ],
  "alerts": [ /* see alerts.json */ ],
  "cards": [ /* see cards */ ]
}
```

```jsonc
// alerts.json  (also embedded in audit.json)
{
  "id": "oos_video03_0007",
  "type": "OUT_OF_STOCK | MISSING_PRICE_TAG",
  "severity": "high|medium",
  "video_id": "video_03",
  "timestamp_s": 41.2,
  "frame_idx": 1030,
  "bbox_xyxy": [x1,y1,x2,y2],          // in original-frame coords
  "evidence_crop": "cards/../oos_0007.jpg",
  "price_tag": { "track_id": 12, "price": 99.9, "name": "...", "barcode": "460..." } | null,
  "product": { "track_id": null, "facing_count": 0 } | { ... },
  "first_seen_s": 40.1, "last_seen_s": 42.6, "persistence_frames": 22
}
```

```jsonc
// a product card
{
  "card_id": "card_video03_0019",
  "best_crop": "cards/card_0019.jpg",
  "facing_count": 3,
  "seen_from_s": 12.0, "seen_to_s": 18.4,
  "matched_price_tag_track_id": 12,
  "price": 99.9, "loyalty_price": 89.9,
  "name": "Молоко ...",            // from matched FinalTag
  "barcode": "460...",
  "catalog": { "matched": true, "canonical_name": "...", "source": "barcode" },
  "embedding_dim": 384 | null
}
```

New `schema.py` additions: `AlertType` enum, `ShelfAlert`, `ProductCard`,
`ShelfAudit` (run-scale), plus a `RelationStatus` literal. Existing types stay.

## 7. Association algorithm (NEW — both directions)

Per-frame single-shot matching (friend's `match_price_tags`) is noisy on robot
video (motion, occlusion, partial views) and only emits tag→group, never the
*unmatched* sets we need for alerts. Replace with **track-level temporal
association**:

1. Group product detections by `track_id` → product tracks (frame→bbox map).
   Group `FinalTag`s (already per price-tag track).
2. For each price-tag track P and product track Q, take the set of frames where
   **both are visible** (`co-frames`). If empty → not associable.
3. **Forward-rotate both boxes into CCW-upright space** (`_to_upright`,
   §5) — mandatory; the geometry is meaningless in original space. Then per
   co-frame compute the friend's `matcher._score`-style geometry (horizontal
   overlap, "product sits above tag", x-centre distance) + optional name/text
   agreement. Aggregate as a robust mean over co-frames (drop the worst 20%).
4. Greedy 1-tag→best-product assignment with an ambiguity margin (reuse
   `MatchingConfig.min_score` / `ambiguous_margin` semantics).
5. Emit:
   - `relations` (status `ok`/`ambiguous`),
   - **`unmatched_price_tags`** → candidate `OUT_OF_STOCK`,
   - **`unmatched_products`** → candidate `MISSING_PRICE_TAG`.
6. **Persistence gate (false-alarm killer).** Only raise an alert if the
   unmatched track persisted ≥ `K` frames / ≥ `T` seconds (a momentary
   occlusion or a single bad frame must not fire). Tunable; default K≈10.

Geometry note — **P0 DONE, confirmed.** After mapping to CCW-upright space the
price tag sits **below** its product (friend's `below_distance` assumption
holds): tag-above-product ratio 1.00 upright vs 0.60 original on `25_12-20`.
Probe: `scripts/shelf_audit_p0_probe.py`.

## 8. Product card / catalog builder (option 3)

- **Identity = product `track_id`** (primary; free from BoT-SORT). One card per
  product track; `facing_count` = max simultaneous sibling facings in a frame
  (cheap, no embedding needed).
- **Best crop**: across the track's frames pick the sharpest (reuse the
  Tenengrad/Laplacian sharpness already used by the price-tag rectifier) and
  largest, well inside the frame. Save to `cards/`.
- **Tag info**: from the associated `FinalTag` (price/loyalty/name/barcode).
- **Catalog canonicalisation**: feed `(barcode, name)` to the existing
  `catalog.reconcile` (db_hack.csv) → canonical name; fill-only, GT-safe policy
  (memory `catalog-reconciliation-worktree`).
- **Embedding (OPTIONAL, off the critical path)**: DINOv2 ViT-S/14 (384-d) to
  merge the same product seen on two separate passes / two tracks. If torch/
  the model is unavailable, fall back to no-merge (track-id only) — cards still
  work. Keep this a clearly-gated optional dependency (memory
  `no-global-pip-use-uv-venv`: install into the uv venv only).

## 9. Notifications

For the hackathon, "notify" = the structured `alerts.json` + the UI alerts feed
(red badges, sorted by severity) + the headline summary number. A real push
(Telegram/email/webhook) is a 20-line adapter over `alerts.json` — documented as
a trivial extension, **not built** unless asked (avoid scope creep).

## 10. UI surface (optional, last phase)

Mirror `frontend/src/pages/PipelinePage.tsx` conventions: new
`ShelfAuditPage.tsx` + route + `api/` method. Two panels: left = alerts feed
(thumbnail, type, timestamp, jump-to-frame); right = product-card gallery
(photo, name, price, barcode, facings). Backend/ml are a mocked skeleton
(`docs/index.md`), so the runner must work **CLI-first** and write static
`outputs/shelf_audit/...`; the page reads that contract. If wiring the live
backend is out of time, a static fixture view still demos fully.

## 11. Phased plan with exit criteria

- **P0 — Detector reality check. ✅ DONE.** `scripts/shelf_audit_p0_probe.py`
  runs both detectors on the same ccw stream. Findings on `25_12-20`: product
  detector produces tight per-bottle facings (conf median 0.59, p25 0.51, min
  0.25, up to ~23/frame); price-tag detector very confident (0.86–0.89, ~5/
  frame); ccw orientation correct; **association must run in CCW-upright space**
  (tag-above-product 1.00 upright vs 0.60 original). Recommended start `conf≈
  0.30` for products (trim the 0.25 tail), final calibration in P4.
- **P1 — Schema + associator. ✅ DONE.** Added `geometry.to_upright`
  (inverse of `unrotate_box_xyxy`), run-scale `schema.py` types (`AlertType`,
  `ShelfAlert`, `ProductCard`, `ShelfAudit`), and `shelf_analytics/associate.py`
  (`TrackTrace` → `associate_tracks` → relations + `unmatched_price_tags` +
  `unmatched_products`, upright-space scoring, trimmed-mean over co-frames,
  persistence gate). Tests `tests/test_shelf_audit_associate.py` (8) +
  existing shelf tests (8) all green; `to_upright` proven inverse of
  `unrotate_box_xyxy` for ccw/cw.
- **P2 — Runner + alerts. ✅ DONE (code), validation-scoped.**
  `scripts/run_shelf_audit.py`: two detector-only passes (product + price-tag,
  base.txt only — no OCR) → `associate_tracks` → `ShelfAudit` +
  `audit.json`/`alerts.json` + evidence crops. CSV/submission path untouched.
  Verified end-to-end on `25_12-20` (25-frame CPU cap): well-formed schema,
  sensible `ok` relations (tag→product 0.77, correct reasons), real evidence
  crops (a clean price-tag for OOS, a clear bottle for missing-tag).
  **Known over-alerting** on the capped run (21 missing-tag / 31 product
  tracks): each bottle is its own facing track while one tag serves a whole SKU
  group, and a tiny window means tags/products barely co-occur. The fix is
  **(a)** group sibling facings into one product before alerting (P3 cards make
  the unit a product, not a facing) and **(b)** full-length runs so tags &
  products co-occur — both are P3/P4. Full all-5 length needs the user's GPU
  (CPU ≈ hours/video at imgsz 1280); folded into P4.
- **P3 — Product cards + catalog (1 day).** Best-crop + tag-info + catalog
  reconcile; cards written. *Exit:* `cards/` gallery JSON correct; barcode/name
  canonicalised.
- **P4 — Validation & tuning (½ day).** Qualitative review on the 5 videos;
  tune `conf` / score thresholds / persistence K; compute counts + a defensible
  est-lost-revenue formula. *Exit:* low obvious false-alarm rate; honest
  numbers (memory `verify-dont-assert`: report failures too, no single-anecdote
  claims).
- **P5 — Optional embedding merge + UI (1–2 days, stretch).** DINOv2 dedup;
  `ShelfAuditPage`. *Exit:* cross-pass product merge; jury-facing page or static
  fixture demo.

Critical path to a demo = **P0→P4** (CLI + JSON + crops). P5 is upside.

## 12. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Model not fine-tuned on Lenta (generic SKU-110K) → noisy boxes | P0 calibrates `conf`/`imgsz`; persistence gate suppresses flicker; fine-tune is an explicit *later* option, not MVP (memory `incremental-not-rewrite`) |
| Wrong orientation → garbage | Reuse proven `rotate=ccw`; P0 visual confirm (memory `detector-needs-upright-frames`) |
| Occlusion → false OUT_OF_STOCK | Persistence gate (≥K frames) + co-frame robust mean |
| Robot motion → tag/product never co-visible | Track-level over the whole pass, not single frame; fall back to nearest-time window |
| Perf (two detectors over video) | Reuse streaming; product detector can sample every Nth frame; run sequentially, document runtime |
| Windows / Cyrillic | `workers=0`-safe path, utf-8 everywhere, Cyrillic-safe paths (memory `no-global-pip-use-uv-venv`, `fix-root-cause-and-explain`); checkpoint abs-path since gitignored |
| No OOS ground truth | Qualitative + counts + honest confidence story; never claim a metric we can't back |

## 13. Out of scope (explicit)

Fine-tuning the product detector; exact SKU recognition; planogram compliance;
real push notifications; merging into the scored CSV; per-image
`build_shelf_state` rework. All deferred or owned elsewhere.
