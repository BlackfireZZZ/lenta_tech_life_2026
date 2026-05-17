# Current ML Architecture State

Snapshot date: 2026-05-17.  
Branch inspected: `feature/full-autonomous-demo`.  
Scope: the ML project under `projects/price_tag_pipeline/`, its data and runtime
interfaces, and the thin deployable ML service under `ml/`.

This document describes what is present in the working tree now. It separates
implemented behavior from stubs, mocks, and planned target architecture.

## 1. Executive Summary

The repository contains two different things that are both called "ML":

1. `projects/price_tag_pipeline/` is the actual ML/research package. It owns
   data ingestion, detector training, video inference, OCR/VLM extraction,
   QR parsing, aggregation, metrics, export scripts, and tests.
2. `ml/` is a deployable FastAPI wrapper intended to expose the pipeline to the
   product backend. It can run the real pipeline when runtime dependencies and
   weights are available, and falls back to a fake CSV when `ML_MOCK=1` or the
   pipeline import fails.

The implemented ML package is a configurable video pipeline for Lenta price-tag
recognition:

```text
video
  -> detector/tracker
  -> padded or perspective crop rectifier
  -> optional QR decode
  -> OCR or VLM extraction
  -> parser into structured fields
  -> per-track crop buffering and per-field voting
  -> cross-track deduplication
  -> JSONL final tags
  -> hackathon CSV export
```

The system is optimized around the hackathon metric: one CSV row per unique tag,
barcode/geometry/timestamp matching, and at least 80% substantive field accuracy
per matched tag.

## 2. Repository Placement

Relevant directories:

```text
ultralytics/
├── projects/price_tag_pipeline/     # ML package: model, pipeline, scripts, tests
│   ├── src/price_tag_pipeline/      # importable Python package
│   ├── scripts/                     # operational CLI entry points
│   ├── configs/                     # runtime profiles
│   ├── requirements/                # split dependency groups
│   └── tests/                       # unit and smoke tests
├── data/                            # canonical local data/checkpoint layout
│   ├── raw/
│   ├── processed/
│   ├── splits/
│   └── checkpoints/
├── ml/                              # FastAPI ML service wrapper, real-or-mock
├── backend/app/ml/                  # backend-side client/schema mirror
└── docs/                            # strategy, task, data, architecture docs
```

There is also data outside the repo root in `/Users/cute/Lenta_Tech/Данные`.
Inside the repo, the canonical layout is `ultralytics/data/`.

## 3. Current Working Tree Notes

After syncing from git, `lenta/main` has been merged into
`feature/full-autonomous-demo`. This snapshot describes the merged working tree,
including the real ML service bridge, progress reporting, and the external data
preparation work from the feature branch.

## 4. Runtime Pipeline

Main entry point: `price_tag_pipeline.pipeline.PriceTagPipeline`.

Constructor dependencies are built from `PipelineConfig`:

- `build_detector(cfg.detector)`
- `build_rectifier(cfg.rectifier)`
- `build_ocr_engine(cfg.ocr)`
- `QRCodeExtractor()`
- `TagParser(cfg.parser)`
- `TrackAggregator(cfg.aggregation)`
- optional `SuperResolution(scale=2)`

The public API is:

```python
PriceTagPipeline(cfg).run(
    video_path: str,
    output_path: str | None = None,
    progress: ProgressLike = None,
) -> list[FinalTag]
```

`run()` streams frames and detections from the detector. For every detection
with a track id it:

1. updates track liveness in the aggregator;
2. rejects low-confidence detections;
3. rectifies the crop;
4. rejects blurry or tiny crops;
5. optionally upscales tiny crops;
6. pushes the crop into a per-track top-quality buffer.

When a track expires by TTL, or at video end, the pipeline runs QR and OCR only
on the best `top_k_crops_per_track` buffered crops. This is a key current design
choice: OCR/VLM cost is spent on the sharpest crops instead of every Nth frame.

Finalization then:

- parses each OCR/VLM result into `ParsedTag`;
- adds `TagObservation` objects to the aggregator;
- performs per-field voting;
- drops low-confidence tracks;
- runs cross-track deduplication;
- optionally writes JSON or JSONL.

## 5. Domain Types and Internal Contracts

Core dataclasses live in `src/price_tag_pipeline/types.py`.

Important types:

- `Detection`: one detector/tracker output for one frame.
- `CropCandidate`: rectified crop plus quality metadata.
- `OCRResult`: text or JSON string returned by OCR/VLM backend.
- `ParsedTag`: structured fields parsed from OCR/VLM/QR.
- `TagObservation`: one crop-level observation attached to a track.
- `FinalTag`: voted and deduplicated final tag, one candidate submission row.

The internal schema has two layers:

1. Core fields:
   - `regular_price`
   - `loyalty_price`
   - `product_name`
   - `weight_value`
   - `weight_unit`
   - `price_per_unit_value`
   - `price_per_unit_unit`
   - `promo_flag`
   - `currency`
2. Hackathon extra fields carried through `extra_fields`:
   - `price_discount`
   - `barcode`
   - `discount_amount`
   - `id_sku`
   - `print_datetime`
   - `code`
   - `additional_info`
   - `color`
   - `special_symbols`
   - all 11 QR output fields

`FinalTag.to_dict()` flattens both layers into a JSON-serializable record.

## 6. Configuration Model

Config loading is in `src/price_tag_pipeline/config.py`. Runtime profiles are
YAML files in `projects/price_tag_pipeline/configs/`.

Config sections:

- `runtime`: profile name, output path, logging cadence, seed, FPS override,
  optional OCR audit path.
- `detector`: backend, model path, confidence/IoU thresholds, tracker YAML,
  classes, open-vocabulary labels, image size. `model_path` can be local,
  an Ultralytics model name, or `hf://owner/repo/path.pt`.
- `rectifier`: padding, CLAHE, rotation, color preservation, optional
  perspective correction, optional super-resolution.
- `ocr`: backend, crop gates, top-K crops, VLM model/prompt settings, vLLM URL,
  guided JSON flag, few-shot prompt examples, ensemble members.
- `parser`: min/max plausible price.
- `aggregation`: min observations, final confidence, track TTL, fuzzy thresholds,
  dedup thresholds.
- `training`: dataset and detector training defaults.

Current profiles:

- `fast.yaml`: OpenFoodFacts YOLO detector, ByteTrack, smaller image size, classical
  PaddleOCR, stricter crop gates.
- `balanced.yaml`: default OpenFoodFacts detector profile, BoT-SORT, PaddleOCR.
- `hq.yaml`: OpenFoodFacts detector + BoT-SORT + PaddleOCR-VL.
- `zeroshot_nolabel.yaml`: YOLO-World open-vocabulary detection with PaddleOCR.
- `local_smoke_zeroshot.yaml`: very permissive local smoke profile using
  `yolov8s-worldv2.pt`.
- `hq_glm_ocr.yaml`, `hq_qwen3_vl.yaml`, `hq_dots_ocr.yaml`,
  `hq_hunyuan_ocr.yaml`, `hq_rolm_ocr.yaml`, `hq_monkey_ocr.yaml`: single VLM
  OCR profiles.
- `hq_vllm.yaml`: OpenAI-compatible local vLLM/SGLang server profile.
- `hq_ensemble.yaml`: multi-VLM ensemble profile with audit output,
  perspective rectification, super-resolution, and few-shot examples.

## 7. Detection and Tracking

Implemented file: `src/price_tag_pipeline/detector.py`.

Current detector factory:

- `yolo`, `ultralytics`, `yolo11`, `yolo26`, `yolo_world`, `yolo-world`
  all instantiate `YOLOTrackerDetector`.
- `rfdetr`, `rf-detr`, `rf_detr` instantiate `RFDETRDetector`, but that class
  raises `NotImplementedError`.

`YOLOTrackerDetector` uses Ultralytics `YOLO.track()` with:

- `source=video_path`
- `stream=True`
- configured `conf`, `iou`, `tracker`, `device`, `imgsz`
- optional `classes`
- optional open-vocabulary labels via `model.set_classes(...)`

Important current behavior:

- FPS is read from the video container through OpenCV instead of hardcoding
  30 FPS. `runtime.fps_override` can override it.
- If Ultralytics returns boxes without track IDs, the code assigns lightweight
  IoU-based fallback IDs.
- If `model.track()` fails, the detector falls back to `model.predict()` plus
  an internal IoU tracker.
- Production profiles default to OpenFoodFacts'
  `hf://openfoodfacts/price-tag-detection/weights/best.pt`, a YOLO11x
  Ultralytics detector trained on Open Prices price-tag images. The first run
  downloads it through `huggingface-hub` unless it is already cached.

Implemented but not integrated into the main `PriceTagPipeline` loop:

- `inference/sahi_adapter.py`: SAHI tiled prediction wrapper and pure-Python
  fallback tiling.
- `inference/tta.py`: identity/flip/multiscale test-time augmentation.
- `inference/wbf.py`: pure-Python Weighted Box Fusion with optional
  `ensemble_boxes` fallback.

Planned or stubbed:

- RF-DETR inference and training are documented but not wired.
- SAHI/TTA/WBF are utilities, not currently used by `detector.py` or
  `pipeline.py`.

## 8. Crop Rectification and Image Quality

Implemented file: `src/price_tag_pipeline/rectifier.py`.

Two rectifier classes exist:

- `TagRectifier`: padded axis-aligned crop, optional 90-degree rotation for
  vertical tags, CLAHE on luminance, color preserved by default.
- `PerspectiveRectifier`: heuristic contour/quadrilateral detection inside the
  padded crop, homography warp to a rectangle, then the same CLAHE/quality path.
  If no clean quad is found, it falls back to `TagRectifier`.

Crop quality:

- sharpness is computed with Laplacian variance via `quality.py`;
- crop area is stored as pixels;
- the pipeline gates crops by `ocr.min_sharpness`, `ocr.min_crop_area_px`, and
  `ocr.min_detection_confidence`.

Optional super-resolution:

- `super_resolution.py` lazily tries Real-ESRGAN;
- if Real-ESRGAN is unavailable, it falls back to bicubic upsampling;
- enabled by `rectifier.super_resolution`.

## 9. OCR, VLM, and QR Extraction

Implemented file: `src/price_tag_pipeline/ocr.py`.

All OCR/VLM backends implement `BaseOCREngine`.

Classical backends:

- `noop`: empty output for CI/smoke paths.
- `tesseract`: `pytesseract.image_to_data`, returns joined words and average
  confidence.
- `paddle`: PaddleOCR 3.x, Russian language by default.

VLM backends:

- `paddle_vl`: PaddleOCR-VL path, lazy-loaded.
- `glm_ocr`: GLM-OCR via Transformers.
- `qwen3_vl`: Qwen3-VL via Transformers.
- `dots_ocr`: dots.ocr via Transformers.
- `hunyuan_ocr`: HunyuanOCR via Transformers.
- `rolm_ocr`: RolmOCR via Transformers.
- `monkey_ocr`: MonkeyOCR via Transformers.
- `intern_vl3`: InternVL3 placeholder subclass over the generic path.
- `transformers_vlm`: configurable generic Hugging Face VLM.

Server backend:

- `vllm_server`: OpenAI-compatible client pointed at a local vLLM/SGLang
  server. It can optionally pass `guided_json` with the internal JSON schema.

Other:

- `ensemble`: runs multiple sub-engines per crop and returns all results.
- `mineru`: declared as a pipeline parser option, but `recognize()` raises
  `NotImplementedError`.

The default VLM prompt requests a single JSON object with visible price-tag
fields and QR fields. A JSON schema for guided decoding exists in code.

QR extraction:

- `src/price_tag_pipeline/qr.py` tries OpenCV `QRCodeDetector`.
- It also tries `pyzbar` when installed.
- It parses JSON, URL query strings, and simple key/value payloads.
- It normalizes common aliases like `b`, `p1`, `wL1C`, `aP`, `aC` into the
  hackathon QR fields.
- If QR barcode is found, it also fills visible `barcode` by default.

## 10. Parsing

Implemented file: `src/price_tag_pipeline/parser.py`.

Two parser entry points:

- `parse_text(text, ocr_confidence, backend)`: for classical OCR output.
- `parse_vlm_json(raw_json, vlm_confidence, backend)`: for structured VLM JSON,
  with fallback to `parse_text()` on invalid JSON.

The text parser handles current Russian price-tag cases:

- superscript kopecks such as `129⁹⁹`;
- comma decimals and thousand separators;
- RUB markers;
- loyalty/promo cue words;
- regular vs loyalty price assignment;
- weight/unit extraction;
- price-per-unit extraction;
- promo flag detection;
- product-name best-effort extraction;
- barcode-like digit extraction.

The VLM JSON parser maps all core fields plus every `HACK_EXTRA_FIELDS` entry
into `ParsedTag`.

## 11. Aggregation and Deduplication

Implemented file: `src/price_tag_pipeline/aggregator.py`.

`TrackAggregator` owns live track state:

- observations;
- buffered crop candidates;
- last seen frame;
- last OCR frame;
- last bbox.

Current crop buffering:

- `push_crop()` computes a quality score from sharpness, area, and detector
  confidence;
- track buffers are capped at 64 entries;
- `best_crops(track_id, k)` returns the top-K crops for OCR.

Track finalization:

- drops tracks with fewer than `min_observations_per_track`;
- votes numeric fields with fuzzy buckets;
- votes categorical fields by weighted confidence;
- votes product names using normalized text similarity;
- votes arbitrary hackathon extra fields separately;
- computes `overall_confidence` from non-zero field confidences;
- drops tracks below `min_final_confidence`.

Voting weight combines:

```text
0.55 * parser_or_vlm_field_confidence
+ 0.25 * detection_confidence
+ 0.20 * normalized_sharpness
```

Cross-track deduplication:

- runs after all tracks are flushed;
- considers time window and bbox IoU;
- requires matching regular price, loyalty price, or similar product name;
- merges connected components and keeps the highest-confidence representative.

## 12. Output Artifacts

Pipeline output:

- `.json` output path writes an array of `FinalTag.to_dict()`.
- every other suffix writes JSONL, one final tag per line.

Hackathon CSV export:

- script: `scripts/export_hack_csv.py`
- input: directory of per-video JSONL files;
- output columns: 29-column hackathon CSV schema;
- `frame_timestamp` is emitted in milliseconds;
- `filename` defaults to the JSONL stem, with optional `--video-ext`;
- `price_discount` is derived from regular minus card price if not already
  present.

Generic submission builder:

- module: `src/price_tag_pipeline/submission.py`
- script: `scripts/build_submission.py`
- builds JSON and/or generic CSV from JSONL predictions;
- validates a lightweight schema around `video_id`, `track_id`, `bbox`, and
  `currency`.

Important current gap:

- `ml/app/runner.py` comments reference `final_tags_to_csv`, but that function
  does not exist in `submission.py` right now. The real implemented CSV path is
  `scripts/export_hack_csv.py` and its local helper functions.

## 13. Data Architecture

Canonical layout is documented in `docs/data/layout.md` and implemented by
`src/price_tag_pipeline/data/`.

Current dataset facts from docs:

- 5 labeled videos:
  - `25_12-20`
  - `25_2-10`
  - `26_12-20`
  - `43_15`
  - `49_5`
- 3 unlabeled videos.
- Organizer format is one folder per video: `{id}/{id}.mp4` plus `{id}.csv`.

Current repo data files include:

- `data/processed/dataset.yaml`
- `data/processed/train_images.txt`
- `data/processed/val_images.txt`
- `data/processed/gt_e2e/*.jsonl`
- `data/splits/fold_0.json` through `fold_4.json`
- detector checkpoint files present locally under `data/checkpoints/detector/`,
  including `yolo11n.pt` and `yolov8s-worldv2.pt`.

Data ingestion:

- `scripts/ingest_real_data.py`: organizer layout to canonical `data/raw/`.
- `scripts/prepare_data.py`: detect raw format, ingest, validate, write
  `dataset.yaml`.
- `scripts/make_splits.py`: video-level K-fold split manifests and
  Ultralytics-compatible train/val lists.

Supported raw formats:

- Lenta hackathon CSV plus videos.
- YOLO labels plus frames/videos.
- COCO JSON.
- frames-only is detected but rejected for training because annotations are
  missing.

Important data rules implemented in code:

- `frame_timestamp` is treated as milliseconds, not frame index.
- non-ASCII still-image paths use `cv_io.imread/imwrite` instead of direct
  `cv2.imread/imwrite`.
- Lenta CSV header typo `wholesale_level_1_coun` is normalized.
- integrity validation fails on missing images/labels, bad boxes, unreadable
  images, and class-id issues.

## 14. Training Architecture

Detector training:

- script: `scripts/train_detector_yolo.py`
- backend: Ultralytics `YOLO.train()`
- default model: `hf://openfoodfacts/price-tag-detection/weights/best.pt`
- default epochs: 200
- default image size: 1280
- default batch: 8
- optional W&B via `--wandb`
- optional heavier Albumentations hook via `--use-albu`

Augmentation:

- `training/augmentation.py`: Ultralytics-native augmentation defaults.
- `training/augmentation_albu.py`: best-effort monkey patch for heavy
  Albumentations inside Ultralytics.

Stubbed training:

- `scripts/train_detector_rfdetr.py`: argument surface exists, then raises
  until `rfdetr` and COCO path are wired.
- `scripts/train_vlm_lora.py`: LoRA fine-tuning arguments exist, then raises
  until crop/JSON training pairs exist and training code is implemented.

The runtime detector and the training script intentionally differ right now:
inference starts from the OpenFoodFacts price-tag detector by default, while new
detector fine-tuning experiments still default to YOLO26-l unless `--model` is
overridden.

The root `Dockerfile` is a CUDA-ready training/inference image. It installs
Python, ffmpeg, OpenCV/Tesseract system dependencies, CUDA Torch wheels, and
the full ML requirements.

## 15. Evaluation and Visualization

Metrics modules:

- `metrics/detection.py`: torchmetrics MeanAveragePrecision wrapper.
- `metrics/ocr_metrics.py`: pure-Python CER/WER.
- `metrics/e2e.py`: aligned-tag field accuracy and per-field report.

Evaluation scripts:

- `scripts/eval_detector.py`: detector evaluation.
- `scripts/eval_e2e.py`: JSONL prediction vs JSONL GT using greedy IoU
  matching.
- `scripts/eval_hack_csv.py`: hackathon CSV vs GT CSV, grouped by filename,
  greedy bbox/timestamp matching, per-tag threshold default `0.80`.

Visualization:

- `scripts/visualize_predictions.py`: final prediction overlay on video,
  optionally with audit events.
- `scripts/visualize_detector_ocr_smoke.py`: untracked smoke visualizer for
  raw YOLO-World detections plus PaddleOCR crops.

Demo UI:

- `scripts/gradio_app.py`: local upload-video flow returning annotated MP4,
  CSV, and summary.

## 16. Deployable ML Service Boundary

Directory: `ml/`.

Current state:

- FastAPI app exposes `GET /health`, `POST /process`, and
  `GET /progress/{job_id}`.
- `GET /health` reports `"mode": "real"` or `"mock"`.
- `POST /process` calls `app.runner.run_pipeline()`.
- `run_pipeline()` returns `_MOCK_CSV` only when `ML_MOCK=1` or importing the
  pipeline fails. Otherwise it loads `ML_PIPELINE_CONFIG` (default
  `configs/balanced.yaml`), calls `PriceTagPipeline.run(..., progress=...)`,
  and converts `FinalTag` rows through `price_tag_pipeline.submission`.

Contract:

```text
POST /process
request:  { "video_path": "...", "job_id": "..." }
response: { "csv": "...", "rows": 0, "meta": {...} }

GET /health
response: { "status": "ok", "service": "ml", "mode": "real"|"mock" }
```

Backend mirror:

- `backend/app/ml/schemas.py` mirrors `ml/app/contract.py`.
- `backend/app/ml/client.py` is the only intended backend-to-ML seam.
- backend client is also mocked while `MOCK_MODE=true`.

Product-level status:

- backend jobs API is mocked with an in-memory store.
- frontend is an upload/poll/review/download SPA wired to the mock backend.
- Postgres, Redis, backend, frontend, and ML service are wired in
  `docker-compose.yaml`. Full real processing still depends on runtime deps,
  mounted weights/cache, and disabling mock mode.

The correct current mental model is:

```text
CLI / Gradio scripts
  -> real price_tag_pipeline package

docker-compose product stack
  -> backend mock/in-memory jobs
  -> ml real-or-mock wrapper
  -> real model execution only when configured deps/weights are present
```

## 17. Dependencies

Split dependency files:

- `requirements/base.txt`: numpy, OpenCV, Pillow, PyYAML, scikit-learn,
  Ultralytics.
- `requirements/ocr.txt`: base plus PaddleOCR/PaddlePaddle, pytesseract,
  Transformers/Torch/VLM helpers, OpenAI SDK for vLLM server path.
- `requirements/train.txt`: base plus Torch/Torchvision/Torchmetrics,
  Albumentations, W&B, PEFT/Transformers datasets for VLM training.
- `requirements/dev.txt`: base plus pytest, pytest-xdist, ruff, black.
- `requirements/demo.txt`: demo/UI dependencies.
- `requirements.txt`: convenience aggregate for full dev/training/demo install.

Imports are intentionally lazy in detector/OCR/SR code so parser, aggregator,
data, and tests can import without every heavy backend installed.

## 18. Test Coverage

Current tests cover:

- data ingestion, validation, split, and Lenta CSV conversion;
- Lenta timestamp parsing and header normalization;
- price parser normalization and Russian price-tag quirks;
- aggregator voting and dedup;
- QR payload parsing;
- WBF fallback;
- metrics;
- generic submission builder;
- smoke-style aggregator round trip.

Tests are designed to avoid requiring real data, trained models, or GPU.

This document did not run the test suite. It records architecture from source
inspection only.

## 19. Implemented vs Stubbed

Implemented:

- config loading;
- YOLO/YOLO-World detector path through Ultralytics;
- fallback IoU tracker;
- FPS reading from video metadata;
- crop rectification, perspective fallback, quality scoring;
- optional super-resolution wrapper;
- QR extraction and payload parsing;
- classical OCR engines;
- multiple VLM engine wrappers;
- vLLM/OpenAI-compatible OCR server client;
- OCR ensemble wrapper;
- text parser and VLM JSON parser;
- track-level crop buffering, voting, finalization, deduplication;
- JSON/JSONL output;
- hackathon CSV export script;
- real-data staging, data preparation, validation, splits;
- YOLO detector training script;
- metrics and evaluation scripts;
- Gradio demo script;
- product service contracts;
- ML service bridge to the real pipeline;
- frontend upload/poll/review/download mock workflow.

Stubbed or mocked:

- RF-DETR detector runtime.
- RF-DETR training script body.
- VLM LoRA training script body.
- MinerU OCR backend.
- backend ML client real HTTP call while `MOCK_MODE=true`.
- backend job persistence and async worker.

Implemented but not wired into the main loop:

- SAHI tiled inference.
- TTA.
- detector WBF.

## 20. Current Architectural Risks

1. Product service and CLI are split. The CLI/Gradio path is the most direct
   real pipeline path; `ml/` can run it too, but product demos still need the
   correct environment variables, deps, cache/weights, and backend mode.
2. The default detector is now an AGPL-3.0 OpenFoodFacts checkpoint. That is a
   useful bootstrap, but final submission should validate license implications
   and replace it with a local Lenta-fine-tuned checkpoint if it wins.
3. RF-DETR is still only a documented target. Current detection relies on
   Ultralytics YOLO or YOLO-World.
4. SAHI/TTA/WBF utilities exist but are not part of the active inference path.
   Small-object recall improvements from those modules are not currently
   realized by `PriceTagPipeline`.
5. Most VLM backends are wrapper implementations around generic Transformers
   loading. They may need model-specific input handling once exercised.
6. `pipeline.py` accesses `aggregator._tracks` directly to find expiring tracks.
   It works, but the boundary between pipeline orchestration and aggregator
   state is not fully encapsulated.
7. The pipeline emits empty strings for unrecognized CSV cells in the export
   path, while the task distinguishes absent fields (`"нет"`) from unrecognized
   fields (empty). Correct `"нет"` behavior depends on OCR/VLM/QR explicitly
   producing that value.
8. The current service stack has no persistence, queue, GPU image, or mounted
   pipeline package in `ml/`. Long-running real video processing still needs the
   async job path described in `docs/architecture.md`.

## 21. Build-Forward Path from Current State

Nearest steps to turn the current architecture into a real service:

1. Validate OpenFoodFacts detector operating point on held-out Lenta videos,
   then fine-tune and replace `detector.model_path` only if the local model wins.
2. Change `ml/Dockerfile` from mock CPU image to a real image that installs the
   pipeline package and the selected runtime dependencies.
3. Replace backend mock ML client with the actual HTTP call.
4. Replace in-memory jobs with persistent jobs and an async processing path.
5. If using small-object mode, wire SAHI/TTA/WBF into the detector
   path deliberately rather than leaving them as side utilities.
6. Preserve the current tests and add service-level tests for
   `video -> ML service -> CSV`.

## 22. Short Architectural Truth

The ML package is already a coherent local inference and experimentation
system. It has real modules for detection, OCR/VLM, QR, parsing, aggregation,
data preparation, training, evaluation, visualization, and CSV export.

The deployable product integration is no longer only a contract skeleton: the
ML service can call the real pipeline. It is still not production-complete
because dependency packaging, persistence, queueing, GPU/runtime setup, and
backend real-mode wiring remain unfinished.
