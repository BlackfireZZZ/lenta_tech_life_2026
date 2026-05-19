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

> **Note.** This whole path is fully automatic — detection, association and
> card generation run end to end with no human-in-the-loop step.

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

**Implemented base-only (`shelf_analytics/cards.py`).** Sibling *facing* tracks
(one per bottle) are collapsed into one **product card** so alerts/UI are
product-level — this is the fix for the P2 over-alerting. Grouping is
union-find over track pairs that are, in CCW-upright space: on the same shelf
band, horizontally adjacent (gap ≤ N·width), contemporaneous (frame ranges
within N), and (when available) colour-histogram similar. Pure-Python +
deterministic; cv2 colour-hist is computed by the runner and *injected* (keeps
`cards.py` testable, off the CSV path). `regroup_missing_price_tag` flags a card
only if **no** member facing is in an `ok` relation. Verified on `25_12-20`
(25-frame cap): 31 facing tracks → 11 cards, missing-tag **28 facing-level → 4
group-level**.

- **Identity** = card group of product tracks; representative = highest
  best-crop sharpness; `facing_count` = members in the group.
- **Best crop**: sharpest (Laplacian-variance) crop per track, saved to
  `cards/<card_id>.jpg`. *UI polish (P5):* crops are in original (90°-rotated)
  space — rotate upright for display.
- **Tag info (price/loyalty/name/barcode), catalog reconcile, est-lost-revenue
  — DEFERRED to integration:** filled from the real recognition pipeline
  (`FinalTag`) when the heavy OCR stack is installed at merge time; card fields
  are `None` until then by design. Owner decision 2026-05-18.
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

## 10. UI surface — ✅ DONE (base-only, static fixture)

`frontend/src/pages/ShelfAuditPage.tsx` + `api/shelfAudit.ts` + route `/shelf`
+ nav "Аудит полки", mirroring `PipelinePage` conventions (Tailwind tokens,
`Card`/`Badge`, Russian copy). For a zero-backend demo the page reads a
**static fixture** under `frontend/public/shelf-audit/<video>/` built
by `scripts/make_shelf_audit_fixture.py` (copies `audit.json`, rotates
card/evidence crops upright for display, writes a video `index.json`
selector) — zero-backend jury demo. Layout: hero + 4 stat cards + the
facing→card collapse note + two panels (left alerts feed with evidence
thumbnails & timestamps, right product-card gallery). `npm run build` passes
(tsc clean, `ShelfAuditPage` chunk emitted). Price/name/barcode show a
"после интеграции OCR" placeholder until enrichment lands. Live-backend wiring
(swap fetch for the gateway API) is the only integration-time UI task left.

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
- **P3 — Product cards. ✅ DONE (base-only); enrichment deferred.**
  `shelf_analytics/cards.py` (`build_card_set` + `regroup_missing_price_tag`),
  wired into the runner with cv2 best-crop + colour-hist appearance. **Fixes
  the P2 over-alerting**: on `25_12-20` (25-frame cap) 31 facing tracks → 11
  product cards, missing-tag 28 (facing) → 4 (group); alerts now product-level
  and carry frame/bbox + card evidence; total alerts 23 → 6. 7 card tests +
  the rest green (23 total). **Deferred to integration** (owner decision): card
  price/name/barcode via real OCR `FinalTag`, catalog reconcile, est-revenue.
- **P4 — Validation & tuning. ✅ DONE on GPU (RTX 4070 Ti, cu124 torch);
  honest result.** Full-length run on all 5 videos. Found the dominant
  error: the ambiguous gate discarded *strong* top matches (0.79–0.89)
  because an adjacent **sibling facing** scored within `ambiguous_margin` on
  dense shelves. Fix = `AssociationConfig.confident_score=0.62`
  (data-driven): accept top-1 when its score is decisive regardless of a
  close runner-up. **Before→after across 5 videos:** OOS false-positives
  **30→10** total (26_12-20 9→0, 49_5 7→2, 43_15 4→1, 25_12-20 5→2),
  ambiguous down (26_12-20 52→32, 49_5 55→41), ok-relations up (26_12-20
  33→53, 49_5 24→38). Spot-checked: override picks the right product;
  runners-up are confirmed sibling facings; no wrong matches introduced.
  **Honest residual:** `MISSING_PRICE_TAG` ≈ unchanged (46/20/20/19/46) —
  NOT association strictness but (a) low price-tag-detector recall on some
  videos (25_12-20: 7 tag tracks vs 72 cards) and (b) product-card
  over-segmentation across the aisle pass (transient cards whose short
  window never contains a detected tag). This is **not a knob** — it needs
  cross-time same-SKU card merging (the deferred DINOv2/appembedding) +
  better tag recall (integration with the real recognition pass). UI now
  honestly labels OOS as the precise signal and "без ценника" as
  *candidates*. est-lost-revenue still deferred (needs OCR prices).
  *Exit met:* full GPU run done, false-alarm root-causes characterised,
  one principled fix landed + measured, residual documented (no
  over-claim).
- **P5 — UI ✅ DONE (base-only); embedding dedup deferred.**
  `ShelfAuditPage` + `api/shelfAudit.ts` + route/nav + fixture builder;
  `npm run build` passes; static fixture demos fully with no backend/GPU.
  Optional DINOv2 cross-pass merge still deferred (off critical path).

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

## 14. Testing TODO

P0–P3 + P5-UI committed; **P4 ran full-length on all 5 videos on GPU
(2026-05-19)**. Status below reflects that run.

**Blocking — GPU (P4): ✅ DONE**
- [x] `run_shelf_audit.py --all` full-length on all 5 videos (GPU). Finished
      clean (exit 0, twice); memory bounded; per-video product/tag track and
      alert counts in §11 P4.
- [x] Eyeballed alerts: OOS crops are real readable lone tags; missing-tag
      crops are real products. False-alarm modes characterised honestly
      (OOS: was association-strictness, now fixed; missing-tag: tag-recall +
      card over-segmentation — see §11 P4).
- [x] Tuned on real runs: landed `confident_score=0.62` (OOS FP 30→10).
      `--product-conf 0.30` / `--persistence 10` kept (sane on full runs;
      `filtered_non_persistent_*` shows the gate working). Other knobs left
      at defaults — not the bottleneck.
- [x] `to_upright` ccw holds on all 5 (every video produced geometry-sane
      `ok` relations + reasons `tag_below_product`). Per-video probe optional.

**Base-only QA**
- [x] Card grouping quality: spot-checked across videos — clean recognisable
      cards (Le Petit Béret, игристое КРЫМ…); override runners-up confirmed
      sibling facings, not different SKUs.
- [x] Associator on real multi-tag video (26_12-20, 85 tags): ambiguous
      32/53, `ok` relations point at the correct product.
- [x] Windows / Cyrillic: full pipeline ran on the Cyrillic-user box; rotated
      detector path (manual cv2 loop) needs no `workers=0`.
- [ ] Runner robustness edges: video with 0 tags / 0 products; bogus-fps
      fallback; non-`.mp4`; re-run overwrite. (Not explicitly tested.)
- [ ] UI **visual** QA in a browser: `cd frontend && npm run dev` → `/shelf`
      (build + `tsc -b` are clean; not yet eyeballed live across the 5-video
      selector / responsive / broken-image).

**Integration-time (when the real big OCR is installed):**
- [ ] `FinalTag` → card join (by IoU + time, NOT track-id across passes);
      fill price/name/barcode; catalog reconcile (fill-only, GT-safe);
      est-lost-revenue formula + sanity.
- [ ] Swap the UI's static `fetch` for the gateway API; keep the fixture as
      the offline fallback.
- [ ] Re-run full pytest + `eval_*` to confirm the scored CSV path is
      untouched by anything here.
