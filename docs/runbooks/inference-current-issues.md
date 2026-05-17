# Current inference issues and fix plan

Date: 2026-05-17.

This note records the current local inference state after running the pipeline
on the hackathon video data under `/Users/cute/Lenta_Tech/Данные`. It separates
actual model inference from original-CSV reference overlays so we do not mix
predictions with ground truth.

## What was checked

- Repository branch: `feature/full-autonomous-demo`.
- Remote state after fetch: local branch is `ahead 3`, `behind 0` relative to
  `lenta/feature/full-autonomous-demo`; current `HEAD` also matches
  `lenta/main`.
- Test suite: `44 passed`.
- Demo/OCR dependencies were installed into `.venv`.
- Smoke video used: `/Users/cute/Lenta_Tech/Данные/43_15/43_15.mp4`.

Generated local artifacts:

- `outputs/hackathon_smoke/43_15_detector_ocr_smoke.mp4`:
  raw YOLO-World candidate detections plus attempted PaddleOCR. This is
  inference/debug output.
- `outputs/hackathon_smoke/43_15_expected_from_original_csv_NOT_INFERENCE.mp4`:
  overlay from the original hackathon CSV. This is an expected/reference view,
  not model inference.
- `outputs/hackathon_smoke/43_15.jsonl`:
  full pipeline output from the current smoke config; currently empty.

## Problems observed

### 1. Detector checkpoint bootstrapping

Older production configs pointed at `data/checkpoints/detector/best.pt`, which
was not available locally. Current production configs now default to
OpenFoodFacts' Hugging Face detector:

```text
hf://openfoodfacts/price-tag-detection/weights/best.pt
```

Effect:

- A fresh checkout has a real price-tag detector path before we fine-tune our
  own model.
- First run needs network access or a pre-populated Hugging Face cache.
- Zero-shot detections are still possible, but confidence is very low.
- The detector finds many candidate regions only after lowering confidence to
  around `0.02`.
- Low confidence creates many false positives, especially product packages,
  QR-like regions, labels, and shelf clutter.

Possible fixes:

- Train or fine-tune a detector on the labeled hackathon frames from
  `data/processed`.
- Use synthetic Lenta-style price tags as the first high-ROI augmentation path.
- Keep one organizer video as validation and tune `conf`, `iou`, `imgsz`, and
  prompts against it.
- Override `detector.model_path` with the final local checkpoint once it beats
  the OpenFoodFacts baseline on held-out Lenta videos.

### 2. YOLO-World sometimes returns boxes without `track_id`

The pipeline used to skip detections whose `track_id` was `None`. On the local
YOLO-World smoke run, that meant detections could exist but never reach crop
buffering or OCR.

Fix applied:

- Added a lightweight IoU fallback tracker in
  `projects/price_tag_pipeline/src/price_tag_pipeline/detector.py`.
- If Ultralytics returns boxes without tracker IDs, the wrapper now assigns
  stable-enough local IDs so detections can flow into the rest of the pipeline.

Remaining risk:

- This fallback is only a pragmatic smoke/debug tracker.
- For final quality, BoT-SORT/ByteTrack with a trained detector should be the
  primary path.

### 3. Classical PaddleOCR returns empty text on small rotated price-tag crops

PaddleOCR was tested not only on detector crops but also on GT bbox crops from
`43_15.csv`. It still returned empty text on the sampled crops, even after
trying rotations and upscaling.

Effect:

- The detector can draw candidate boxes.
- OCR observations are not produced.
- The full pipeline has no parsed fields to aggregate, so final JSONL remains
  empty.

Likely causes:

- Price tags are small, sideways, motion-blurred, and partially occluded by
  shelf rails.
- The classical OCR text detector is not robust enough for these crop shapes.
- The crop is often a full tag with tiny text rather than a rectified text-line
  region.

Possible fixes:

- Use a VLM OCR backend for crops, preferably `paddle_vl`, `glm_ocr`,
  `qwen3_vl`, or a local `vllm_server` profile.
- Add a crop rectification stage that estimates tag orientation and warps the
  tag into a horizontal view before OCR.
- Split tag OCR into regions: big price area, product-name area, QR/barcode
  area, secondary fields.
- Add super-resolution only after rectification, not as a blind resize.
- Use the original CSV bboxes to build a crop-level OCR validation set and
  measure which OCR backend actually reads Russian fields.

### 4. Full pipeline output is empty under the current zero-shot smoke config

The smoke config produces candidate detections, but final tags are still zero.
This is not a successful end-to-end inference result.

Current root cause:

- Detection candidates are weak and noisy.
- OCR produces no reliable parsed fields.
- Aggregation has no useful observations to vote on.

Possible fixes:

- First make OCR work on GT crops; do not tune tracking/aggregation before that.
- Then run detector-only evaluation and tune detector thresholds.
- Then tune aggregation thresholds once detector and OCR both produce signals.
- Add debug counters to the pipeline:
  `detections -> crops accepted -> QR reads -> OCR calls -> non-empty OCR ->
  parsed observations -> finalized tags`.

### 5. QR/barcode should be treated as P0, but crop quality is not enough yet

The task metric relies heavily on barcode matching. The current local run sees
QR-like regions, but robust decode from the full moving-video crops is not yet
verified.

Possible fixes:

- Run QR/barcode extraction directly on GT tag crops to get a baseline.
- Add multi-scale QR attempts on rectified crops.
- Add barcode-specific datasets from `docs/data/datasets-research.md`
  (`BarBeR`, ABBYY barcode benchmark) to harden barcode detection/decoding.
- Prefer QR/barcode payload fields over OCR text when both are available.

### 6. CPU inference is slow with low zero-shot thresholds

Lowering the detector confidence to make YOLO-World find tags greatly increases
candidate count. PaddleOCR on many crops is slow on CPU.

Possible fixes:

- Do not OCR every candidate; rank by detector confidence, crop size, sharpness,
  and tag-like aspect ratio.
- Use frame sampling and track-level best-crop selection after detector quality
  improves.
- Cache OCR results during debugging.
- Run VLM/OCR on GPU or a local server profile for serious experiments.

### 7. Reference overlays from original CSV must stay clearly marked

The original hackathon CSV is useful for visual QA, but it is not model output.

Fix applied:

- A separate reference video was created with a clear banner:
  `EXPECTED FROM ORIGINAL CSV / NOT INFERENCE`.

Rule going forward:

- Never present original-CSV overlays as inference.
- Keep inference/debug videos and expected/reference videos separate by file
  name and on-video banner.

## How to use the additional datasets

The dataset landscape is documented in
`docs/data/datasets.md` and `docs/data/datasets-research.md`. The key point is
that no single public dataset solves the whole problem, so external data should
be used module-by-module.

### Detector: price tags first, products second

Goal:

- Get stable price-tag boxes before trying to read text.

Useful datasets:

- Labeled organizer frames from `data/processed`.
- Synthetic Lenta-style tags from templates.
- Roboflow price-tag datasets listed in `docs/data/datasets.md`.
- SKU-110K only as retail-shelf visual pretraining, not as the final class
  definition.

How to use:

- Train the first detector on one class: `price_tag`.
- Mix real organizer frames with synthetic tags composited onto shelf/product
  backgrounds.
- Use SKU-110K for dense shelf pretraining only if it improves feature
  robustness; do not let product boxes become the target output for the graded
  CSV.
- Keep product/facing detection as a separate model or separate head so it does
  not pollute price-tag detection.

Validation:

- Validate on held-out organizer videos only.
- Track false positives on products, shelf rails, and QR-like product labels.

### OCR: build a crop-level curriculum

Goal:

- Make OCR work on price-tag crops before tuning full video aggregation.

Useful datasets:

- Original organizer CSV bboxes as GT crops.
- Synthetic Lenta-style tags with exact field positions.
- RusTitW for Russian text in the wild.
- OCR-Cyrillic-Printed-6/8 for printed Cyrillic pretraining.
- Receipt/text IE datasets such as KORIE only as an architecture reference, not
  as a direct domain match.

How to use:

- Export GT crops from the five organizer CSVs and store expected fields next to
  each crop.
- Render synthetic RU tags with the same field layout and font/print artifacts:
  rotation, blur, glare, compression, shelf-rail occlusion, motion blur.
- Fine-tune or benchmark OCR/VLM backends on this crop set.
- Evaluate field-level extraction, not just OCR character accuracy:
  `price_default`, `price_card`, `barcode`, `product_name`,
  `discount_amount`, `id_sku`, QR fields.

Practical path:

- Start with local VLM OCR profiles (`paddle_vl`, `glm_ocr`, `qwen3_vl`, or
  `vllm_server`) because classical PaddleOCR failed on sampled GT crops.
- Use synthetic tags to teach layout and values.
- Use real GT crops to tune prompts, crop rectification, and confidence policy.

### Barcode and QR: separate P0 module

Goal:

- Maximize barcode/QR extraction because barcode is the primary match key in
  the metric.

Useful datasets:

- BarBeR barcode benchmark.
- ABBYY barcode detection benchmark.
- Organizer GT crops with QR/barcode fields from CSV.

How to use:

- Train or tune a barcode/QR detector separately from general price-tag
  detection.
- Run barcode decoding on the whole tag crop, QR sub-crop, and multiple scaled
  versions.
- Prefer QR/barcode payload values over OCR text when available.
- Add explicit metrics for barcode exact match and partial EAN recovery.

Validation:

- Report barcode recall on GT crops before measuring full CSV score.
- Treat a row with missing barcode as high-risk even if other OCR fields look
  plausible.

### Product/facings/OOS: keep out of the graded CSV path

Goal:

- Build the optional shelf-analytics layer without corrupting the graded
  price-tag CSV.

Useful datasets:

- SKU-110K for dense product/facing detection.
- Locount for localization plus counting.
- SHARD/SHAPE for shelf rows, SKU recognition, and planogram-style reasoning.
- Gap/ROSCH-style datasets for empty-slot/OOS ideas.

How to use:

- Train product/facing detection as a separate output artifact:
  shelf-state JSON/CSV, not the 29-column hackathon CSV.
- Link price tags to nearby products only after price-tag detection and OCR are
  reliable.
- Use temporal tracking across robot video to stabilize facings and OOS
  estimates.

Rule:

- Shelf analytics is a demo/business feature, not a scoring path. It should be
  physically separate from the final graded submission.

### Dataset ingestion plan

Recommended order:

1. Create `data/processed/gt_crops/` from organizer CSV bboxes.
2. Add a synthetic Lenta-tag generator and export YOLO + OCR labels.
3. Add barcode/QR crop benchmark from organizer tags plus BarBeR/ABBYY.
4. Add small Roboflow price-tag datasets only after checking licenses.
5. Add SKU-110K/Locount/SHARD only for pretraining or separate shelf analytics.

Do not download large external datasets blindly. Every dataset should enter the
repo through a small manifest that records source URL, license, task, class
mapping, and whether it is allowed for training, validation, or demo only.

## Recommended next order of work

1. Build a GT-crop OCR benchmark from the organizer CSV bboxes.
2. Test VLM OCR backends on those GT crops and choose the first backend that
   reliably extracts price, barcode, and product name.
3. Train or fine-tune the detector on available labeled frames plus synthetic
   Lenta-style tags.
4. Add pipeline debug counters so empty outputs explain themselves.
5. Re-run full inference on all five labeled videos and compare against the
   original CSV with `eval_hack_csv.py`.
6. Only after the graded CSV path works, revisit shelf analytics/facings/OOS as
   a separate non-metric layer.

## Current bottom line

The repository runs, dependencies install, tests pass, and raw zero-shot
detection can be visualized. The current blocker is model quality, especially
OCR on small rotated Lenta price tags. The fastest path is not more aggregation
tuning; it is crop-level OCR validation plus a trained detector checkpoint.
