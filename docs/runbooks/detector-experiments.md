# Detector experiments runbook (refined plan)

Date: 2026-05-17. Owner of this round: detector quality only — get
**high price-tag recall with few false positives** so OCR has clean crops.
Everything after `detect ->` (QR/OCR/parser/aggregation) is **out of scope for
this round** and unchanged; it stays as written in the friend's
`ml-quality-improvement-plan.md`.

This is the executable, repo-grounded version of that plan. Where I changed
the friend's plan, the reason is stated under **[refinement]**.

## 0. Hard reality of the data (drives every decision)

- **5 videos, ~61 labeled frames, 274 boxes, 1 class `price_tag`.** Frames were
  extracted by `prepare_data.py` **only at annotated timestamps** — the
  detector only ever sees frames that have labels.
- Per-video labeled frames: `49_5`=25, `26_12-20`=15, `25_12-20`=10,
  `25_2-10`=9, `43_15`=**2**.
- Source video is 3840x2160 ~20 FPS → tags are physically small after the
  network downscales the frame.

**[refinement] Validation must be leave-one-video-out, reported aggregated.**
The friend's plan says "held-out organizer video". With 5 videos a single fold
val = 1 video; mAP on 2–25 frames is noisy and `43_15` (2 frames) is too thin
to trust as a val fold on its own. So:

- Train/eval across **all 5 folds** (`data/splits/fold_{0..4}.json`, each holds
  out one video) and report **mean ± spread** of recall/precision, not one
  number. Treat the `43_15`-held-out fold as a smoke signal, not a gate.
- **Never** pick a checkpoint by train loss or even val mAP alone. Visual QA on
  rendered frames is a hard gate (the user explicitly asked for this).
- "Recall on a held-out video" only covers that video's *labeled* frames.
  False-positive discovery needs inference on **un**labeled frames + the 3
  unlabeled videos → that is the hard-negative loop (§5), not the val metric.

## 1. Environment (uv, this worktree)

Lives in the `worktree-experiments` worktree, isolated venv:

```
.claude/worktrees/experiments/.venv      # uv venv, Python 3.12
                              setup_env.log
```

- `torch`/`torchvision` = CUDA cu124 build (RTX 4070 Ti, 12 GB, Ada sm_89).
- Rest from `projects/price_tag_pipeline/requirements/train.txt`.
- Run everything with `.venv/Scripts/python.exe` from the worktree root.
- `data/processed/dataset.yaml` carries an **absolute** `path:` to the
  main-repo `data/processed`, so training from the worktree reads the one real
  dataset. `runs/` land in the worktree (gitignored).

## 2. Stage gate table (kept from the friend's plan)

The friend's §3 table stands. For **this round** only the detector rows are
gated; downstream rows are recorded if cheap, never gated yet:

| Stage | Metric | P0 target | Stretch |
|---|---|---:|---:|
| Detector | `recall@0.5` (LOVO mean) | ≥ 0.90 | ≥ 0.95 |
| Detector | `precision@0.5` (LOVO mean) | ≥ 0.50 | ≥ 0.75 |
| Detector | `avg_det_per_frame` | bounded, inspected | stable per aisle |
| Detector | `FP taxonomy` | top FP classes named | rails/packages rare |

**[refinement]** Recall is the primary objective but precision is *not*
free: every FP becomes an extra OCR call and a dedup/duplicate risk in the
graded CSV. So we optimize **recall first, then claw back precision via hard
negatives**, and we *quantify* FPs by visual class, not just a number.

## 3. Experiment ledger (new)

**[refinement]** The friend's plan says "every run comparable" but defines no
mechanism. Every run appends one row to
`runs/ledger.csv` via `scripts/exp_ledger.py` (added in this round):

```
run_id, model, imgsz, aug, fold, epochs, recall, precision, map50, map5095,
avg_det_per_frame, notes, weights_path
```

QA renders go to `runs/<run_id>/qa/*.jpg` (pred boxes green, GT boxes red).

## 4. Detector training plan

### 4.1 Smoke (verify GPU + pipeline)

```bash
.venv/Scripts/python.exe projects/price_tag_pipeline/scripts/train_detector_yolo.py \
  --dataset data/processed/dataset.yaml --model yolo11n.pt \
  --epochs 15 --imgsz 960 --batch 16 --device 0 --name smoke_n_fold0
```

Pass condition: trains on CUDA, produces `weights/best.pt`, eval + QA render
run without error. Not a quality signal.

### 4.2 Baseline (built-in aug)

**[refinement] Default model = `yolo11s`, not n→s→m blindly.** With 274 boxes a
large head overfits; a COCO-pretrained `yolo11s` backbone + heavy aug is the
sweet spot. `yolo11m` only if `s` clearly underfits recall. `yolo11n` is smoke
only. (Repo defaults mention `yolo26l`; `yolo11` is the safest well-supported
family — revisit yolo26 only after the data loop is exhausted, per friend §12.)

```bash
.venv/Scripts/python.exe projects/price_tag_pipeline/scripts/train_detector_yolo.py \
  --dataset data/processed/dataset.yaml --model yolo11s.pt \
  --epochs 200 --imgsz 1280 --batch 8 --device 0 \
  --patience 40 --name s_builtin_imgsz1280_fold0
```

### 4.3 Levers to sweep (in priority order)

1. **`imgsz` 1280 vs 1536** — 4K source → tags tiny; expect 1536 to lift
   recall most. Biggest single lever here.
2. **`mosaic` 0.5 vs 0.3 vs 0.0** — **[refinement]** `UltralyticsAugConfig`
   itself warns "mosaic hurts small-object recall above ~0.3"; with tiny tags
   this likely matters more than blur aug. Add `close_mosaic` for last epochs.
3. **`--use-albu` heavy profile** vs built-in — domain blur/glare/occlusion
   (the friend's §P0.4). Only kept if LOVO recall or downstream crop
   readability improves (acceptance rule unchanged).
4. **inference `conf` sweep** 0.05 / 0.08 / 0.10 / 0.15 / 0.20 — recall-first.

Each lever changes one variable vs the baseline; compare via the ledger +
visual QA. Keep a change only if held-out (LOVO) recall holds/improves **and**
downstream crop readability does not regress.

### 4.4 yolo26 note

`train_detector_yolo.py` defaults to `yolo26l.pt`. **[refinement]** Defer
yolo26 until §4.3 + §5 are exhausted (matches friend §12: "RF-DETR/yolo26
before YOLO + data loops are exhausted = waste"). yolo11 is the baseline family.

## 5. Hard-negative + active-label loop (highest data leverage)

Friend's §P0.2, made concrete:

1. Run the best detector at low `conf` on the **3 unlabeled videos** and on the
   unlabeled frames of the 5 labeled ones.
2. Render annotated frames; **visually** bucket false positives:
   product packages, shelf rails/edges, promo posters/wobblers, logos,
   reflections/glare, QR-like product graphics.
3. Add FP-heavy frames as hard negatives (image kept, no/empty label) and add
   any missed true tags as new positive labels.
4. Retrain; compare LOVO recall + FP taxonomy counts + annotated videos.

This is the single biggest quality lever given only 274 boxes (consistent with
the docs: tiny set → external/synthetic/hard-neg data is the main lever).
Synthetic Lenta-style tags (friend §10 Tier 0) are the next step **after** this
loop shows diminishing returns — not started this round.

## 6. Definition of done (this round)

1. `balanced.yaml` points at a **trained** local detector, not zero-shot.
2. Every run has a ledger row + QA renders.
3. LOVO-mean `recall@0.5` ≥ 0.90 at IoU 0.5 (stretch 0.95).
4. FP classes named from real annotated frames; top classes mitigated via
   hard negatives.
5. Inference `conf` chosen from the sweep, recall-first, FP cost quantified.

Downstream (QR/OCR/parse/aggregate/CSV) DoD items from the friend's plan are
explicitly **not** part of this round.
