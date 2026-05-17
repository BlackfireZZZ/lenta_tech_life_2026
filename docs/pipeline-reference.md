# Price Tag Video Pipeline — Lenta Tech Life 2026

End-to-end pipeline for detecting price tags in robot-captured retail-store video
and extracting structured fields per tag (regular_price, loyalty_price,
product_name, weight, price-per-unit, promo_flag).

For the documentation map start at [`index.md`](./index.md). The official task
spec is [`hackathon/task.md`](./hackathon/task.md) and the strategic chat intel
is [`hackathon/briefing.md`](./hackathon/briefing.md). For our analysis of the
original scaffold and the rewrite rationale, see [`analysis.md`](./analysis.md).
For the model-selection and training strategy, see [`strategy.md`](./strategy.md).

## Directory layout

```
projects/price_tag_pipeline/
├── configs/                       # runtime profiles
│   ├── fast.yaml                  #   small detector, classical OCR
│   ├── balanced.yaml              #   default
│   └── hq.yaml                    #   larger detector + PaddleOCR-VL
│   └── zeroshot_nolabel.yaml      #   no-label zero-shot baseline (YOLO-World)
├── requirements/                  # split installs
│   ├── base.txt                   #   inference only
│   ├── ocr.txt                    #   adds PaddleOCR / VLM
│   ├── train.txt                  #   adds torch + wandb + albumentations
│   ├── demo.txt                   #   adds Gradio + optional barcode decoder
│   └── dev.txt                    #   adds pytest + ruff + black
├── scripts/                       # CLI entry points
│   ├── prepare_data.py            #   raw -> processed (idempotent)
│   ├── make_splits.py             #   video-level GroupKFold manifests
│   ├── train_detector_yolo.py     #   Ultralytics YOLO training (primary)
│   ├── train_detector_rfdetr.py   #   RF-DETR training (stub, lands with data)
│   ├── train_vlm_lora.py          #   PaddleOCR-VL LoRA fine-tune (stub)
│   ├── eval_detector.py           #   mAP on a fold
│   ├── eval_e2e.py                #   per-field + overall accuracy on a video
│   ├── run_inference.py           #   video -> JSONL tags
│   ├── run_batch_inference.py     #   all videos dir -> per-video JSONL
│   ├── export_hack_csv.py         #   JSONL -> CSV in hackathon schema
│   ├── eval_hack_csv.py           #   CSV-vs-CSV evaluator (>=80% per-tag score)
│   ├── visualize_predictions.py   #   JSONL + video -> annotated MP4
│   └── gradio_app.py              #   upload video -> annotated MP4 + CSV UI
├── src/price_tag_pipeline/
│   ├── aggregator.py              # per-track per-field voting + cross-track dedup
│   ├── cli.py
│   ├── config.py
│   ├── detector.py                # YOLO backend (RF-DETR stub)
│   ├── ocr.py                     # PaddleOCR-VL / PaddleOCR / Tesseract / NoOp
│   ├── parser.py                  # raw text -> structured ParsedTag
│   ├── pipeline.py                # main inference loop
│   ├── qr.py                      # QR/barcode payload extraction
│   ├── quality.py                 # sharpness metric
│   ├── rectifier.py               # padded crop + CLAHE on luminance (keeps color)
│   ├── types.py                   # domain types (ParsedTag, FinalTag, ...)
│   ├── data/                      # YOLO/COCO loaders, validate, splits
│   ├── metrics/                   # detection mAP, OCR CER/WER, E2E field accuracy
│   └── training/                  # augmentation defaults
└── tests/
    ├── test_aggregator.py
    ├── test_price_parser.py
    ├── test_metrics.py
    ├── test_data_pipeline.py
    └── test_pipeline_smoke.py
```

## Quick start

```bash
# 1. Install (Python 3.11 or 3.12 recommended).
pip install -r projects/price_tag_pipeline/requirements.txt

# 2. Drop data under data/raw/ (see docs/data/layout.md for the exact layout).

# 3. Stage frames + labels + run integrity checks.
python projects/price_tag_pipeline/scripts/prepare_data.py \
    --raw data/raw \
    --processed data/processed

# 4. Build a 5-fold video-level split.
python projects/price_tag_pipeline/scripts/make_splits.py \
    --processed data/processed \
    --metadata  data/raw/metadata.csv \
    --out       data/splits \
    --n_splits  5 \
    --emit-dataset-yaml-fold 0

# 5. Train the detector on fold 0.
python projects/price_tag_pipeline/scripts/train_detector_yolo.py \
    --dataset data/processed/dataset.yaml \
    --model   hf://openfoodfacts/price-tag-detection/weights/best.pt \
    --epochs  200 \
    --imgsz   1280 \
    --batch   8 \
    --device  0 \
    --name    openfoodfacts_lenta_fold0 \
    --wandb

# 6. Evaluate the best checkpoint.
python projects/price_tag_pipeline/scripts/eval_detector.py \
    --weights runs/lenta/openfoodfacts_lenta_fold0/weights/best.pt \
    --dataset data/processed/dataset.yaml

# 7. Inference on a held-out video.
python projects/price_tag_pipeline/scripts/run_inference.py \
    --video  path/to/video.mp4 \
    --config projects/price_tag_pipeline/configs/balanced.yaml \
    --output outputs/video01.jsonl

# 8. End-to-end accuracy on a held-out video.
python projects/price_tag_pipeline/scripts/eval_e2e.py \
    --pred outputs/video01.jsonl \
    --gt   data/processed/gt_e2e/video01.jsonl

# 9. Batch inference for all videos + hackathon CSV export.
python projects/price_tag_pipeline/scripts/run_batch_inference.py \
    --videos-dir data/raw/videos \
    --config projects/price_tag_pipeline/configs/zeroshot_nolabel.yaml \
    --outputs-dir outputs/jsonl

python projects/price_tag_pipeline/scripts/export_hack_csv.py \
    --inputs outputs/jsonl \
    --out-csv submission/hack_submission.csv

# 10. Render visual QA video with boxes and recognized fields.
python projects/price_tag_pipeline/scripts/visualize_predictions.py \
    --video data/raw/videos/video01.mp4 \
    --pred outputs/jsonl/video01.jsonl \
    --out outputs/vis/video01_annotated.mp4

# 11. Launch local upload-video UI.
python projects/price_tag_pipeline/scripts/gradio_app.py \
    --config projects/price_tag_pipeline/configs/zeroshot_nolabel.yaml \
    --outputs-dir outputs/demo
```

## Profiles

- `fast.yaml`   — OpenFoodFacts detector, classical PaddleOCR, low TTL. Use for quick smoke runs.
- `balanced.yaml` — production default; OpenFoodFacts detector + BoT-SORT + PaddleOCR.
- `hq.yaml`     — OpenFoodFacts detector + PaddleOCR-VL 1.5 VLM. Best quality, slowest.
- `zeroshot_nolabel.yaml` — no training/labels required (YOLO-World + OCR baseline).

Production profiles use OpenFoodFacts'
`hf://openfoodfacts/price-tag-detection/weights/best.pt` by default. Override
`detector.model_path` in the YAML to use a local fine-tuned checkpoint.

## Backends

### Detector
- `backend: yolo` — Ultralytics (YOLO11 / YOLO12 / YOLO26). Default.
- `backend: rfdetr` — Roboflow RF-DETR (ICLR 2026, current COCO SOTA). Stub;
  lands once we install `rfdetr` and convert data to COCO.

### OCR / structured extraction (May 2026 SOTA matrix)

| Backend           | Params | OmniDocBench v1.5 | Russian | Notes |
|-------------------|--------|-------------------|---------|-------|
| `glm_ocr`         | 0.9B   | 94.62 (#1)        | yes     | Z.AI, Apache-2.0, vLLM/SGLang ready |
| `paddle_vl`       | 0.9B   | 94.50             | yes     | Tied for top; default in `hq.yaml` |
| `mineru` *(stub)* | 1.2B   | 90.67             | yes     | Pipeline-based; not recommended for crops |
| `dots_ocr`        | 3B     | 88.41             | yes     | SOTA multilingual; vLLM-integrated |
| `monkey_ocr`      | 3B     | top of 3B class   | yes     | Beats GPT-4o / Qwen2.5-VL-72B / InternVL3-78B |
| `qwen3_vl`        | 4–235B | strong            | yes     | 201 languages, 256K ctx, best fine-tune story |
| `hunyuan_ocr`     | 1B     | multiple SOTA     | yes     | Tencent; lightweight |
| `rolm_ocr`        | 7B     | strong on olmOCR-Bench | yes | Qwen2.5-VL fine-tune by Reducto |
| `intern_vl3`      | 1–78B  | varies            | yes     | OpenGVLab; multi-tile preprocessing |
| `paddle`          | small  | n/a (classical)   | yes     | Classical PaddleOCR 3.x; fast |
| `tesseract`       | tiny   | n/a (classical)   | partial | Last-resort fallback |
| `noop`            | —      | —                 | —       | Tests / CI only |

### Production server: `vllm_server`
Run a vLLM (or SGLang) server with ANY of the models above and point the
pipeline at `http://localhost:8000/v1`. Model swap is config-only.

```bash
vllm serve zai-org/GLM-OCR --port 8000 --max-model-len 8192
# or:
vllm serve PaddlePaddle/PaddleOCR-VL --port 8000
vllm serve Qwen/Qwen3-VL-32B-Instruct --port 8000 --tensor-parallel-size 2
```

Then use `configs/hq_vllm.yaml`. Ready-to-go per-backend configs:
`hq.yaml` (PaddleOCR-VL), `hq_glm_ocr.yaml`, `hq_qwen3_vl.yaml`,
`hq_dots_ocr.yaml`, `hq_hunyuan_ocr.yaml`, `hq_rolm_ocr.yaml`,
`hq_monkey_ocr.yaml`, `hq_vllm.yaml`.

## Tracking

- ByteTrack (`bytetrack.yaml`) — fast profile.
- BoT-SORT (`botsort.yaml`) — balanced/hq, with camera motion compensation.

## Progress reporting

A robot video is minutes long; the task spec mandates a UI progress bar
([`hackathon/task.md`](./hackathon/task.md) §13) and the product needs a
poll-able job status ([`architecture.md`](./architecture.md) §3.4/§5.3).
Both consume one signal emitted from one place — the inference loop.

- **API.** `PriceTagPipeline.run(video, output_path=None, progress=None)`.
  `progress` is optional and backwards compatible: `None` is a silent
  no-op (existing callers are unaffected); pass a `ProgressReporter` **or**
  a bare `callable(ProgressEvent)`.
- **Module.** `price_tag_pipeline.progress`: `ProgressEvent`
  (`phase`, `fraction` 0..1 monotonic, `frames_done/total`,
  `tags_finalized`, `message`), phases `detect → finalize → dedup → done`,
  and reporters `NullProgress` / `CallbackProgress` / `TqdmProgress`
  (+ `as_reporter()` coercion). A failing reporter is disabled, never
  fatal; detection owns 0..0.97 of the bar, the finalize+dedup tail the
  rest. `tqdm` is a soft dependency (`requirements/base.txt`) — absent, the
  bar degrades to log lines.
- **CLI.** `run_inference.py` / `run_batch_inference.py` show a terminal
  bar by default; disable with `--no-progress`.
- **Gradio.** `gradio_app.py` streams it into a live `gr.Progress` bar
  (pipeline 0–85 %, CSV 85–92 %, annotated render 92–100 %) — this is the
  mandated upload→progress→download UI.
- **ML service.** Recorded into an in-process registry keyed by `job_id`
  and exposed by the additive `GET /progress/{job_id}`
  ([`architecture.md`](./architecture.md) §5).

## Tests

```bash
pip install -r projects/price_tag_pipeline/requirements/dev.txt
pytest projects/price_tag_pipeline/tests -v
```

Unit tests cover: parser (Russian formats), aggregator (multi-field voting + dedup),
metrics (CER/WER/E2E), data pipeline (synthetic fixtures), and pipeline smoke flow.
They do not require a trained model, real data, or GPU.

## Output schema

Per tag (one line of `outputs/*.jsonl`):

```json
{
  "track_id": 37,
  "bbox": [412, 221, 520, 278],
  "timestamp_s": 12.43,
  "source_frames": [372, 376, 380],
  "regular_price": 129.99,
  "loyalty_price": 99.99,
  "product_name": "Молоко Простоквашино 2.5% 1л",
  "weight_value": 1.0,
  "weight_unit": "л",
  "price_per_unit_value": 99.99,
  "price_per_unit_unit": "руб/л",
  "promo_flag": true,
  "currency": "RUB",
  "barcode": "4601234567890",
  "qr_code_barcode": "4601234567890",
  "price1_qr": 129.99,
  "action_price_qr": 99.99,
  "action_code_qr": "A12345",
  "field_confidences": {"regular_price": 0.92, ...},
  "overall_confidence": 0.86,
  "n_observations": 5
}
```

### Graded 29-column CSV

Within the pipeline the graded schema lives in **one** place:
`price_tag_pipeline.submission`.

- `final_tags_to_csv(tags, filename) -> str` — the canonical renderer.
  `list[FinalTag] → final_tags_to_csv` is what the ML service calls
  ([`architecture.md`](./architecture.md) §5); also used in-process by
  `gradio_app.py`.
- `HACK_CSV_COLUMNS` / `hack_row_from_tag_dict()` — the 29-column order and
  the per-tag mapping. `export_hack_csv.py` is now a thin JSONL-replay CLI
  that **delegates** to these (no duplicated schema).

This is the **producer** of the project-wide unifying contract: the same
29 columns, order and byte format (UTF-8, `,` separator, `.` decimal,
`\n` line terminator, `QUOTE_MINIMAL`) are mirrored by the gateway
(`backend/.../schemas/job.py:CSV_COLUMNS` + `jobs_mock.py:build_csv`) and
consumed by the SPA review screen. The three owners must stay in lock-step
— see [`architecture.md`](./architecture.md) §5.6;
`tests/test_submission.py` enforces the column + `\n`/`QUOTE_MINIMAL`
parity on this side.

Fields not recognized stay empty; if OCR/VLM explicitly returns `нет`, the
CSV keeps `нет` (the renderer passes values through — the `нет`-vs-empty
decision is made upstream in the parser/aggregator, never here).
