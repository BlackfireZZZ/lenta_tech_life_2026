# Lenta Tech Life Hackathon Runbook

This is the practical runbook for `feature/full-autonomous-demo`.

The target flow is:

```text
robot video
  -> open-vocabulary price-tag detection
  -> tracking + best-crop selection
  -> QR/barcode extraction
  -> OCR/VLM structured extraction
  -> per-track voting + dedup
  -> annotated MP4 + hackathon CSV
```

## 1. Install

Recommended Python: 3.11 or 3.12.

```bash
git clone git@github.com:BlackfireZZZ/lenta_tech_life_2026.git
cd lenta_tech_life_2026
git switch feature/full-autonomous-demo

python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r projects/price_tag_pipeline/requirements/demo.txt
```

If `pyzbar` complains about `zbar`, the pipeline still runs with OpenCV QR
decoding. Install system zbar only when you want stronger 1D barcode decoding.

## 2. Put Data

```bash
mkdir -p data/raw/videos outputs/jsonl outputs/vis submission
cp /path/to/public/videos/*.mp4 data/raw/videos/
```

Model weights are intentionally not committed. The zero-label baseline can use
the open-vocabulary model configured in:

```text
projects/price_tag_pipeline/configs/zeroshot_nolabel.yaml
```

For trained detector runs, put the detector checkpoint at:

```text
data/checkpoints/detector/best.pt
```

and use `balanced.yaml` or `hq.yaml`.

## 3. Batch Inference

```bash
python projects/price_tag_pipeline/scripts/run_batch_inference.py \
  --videos-dir data/raw/videos \
  --config projects/price_tag_pipeline/configs/zeroshot_nolabel.yaml \
  --outputs-dir outputs/jsonl \
  --log-level INFO
```

Outputs:

- `outputs/jsonl/{video}.jsonl` with one final unique tag per line.
- If the config contains `runtime.audit_path`, an OCR/QR audit JSONL is also produced.

For a single video with explicit audit:

```bash
python projects/price_tag_pipeline/scripts/run_inference.py \
  --video data/raw/videos/video01.mp4 \
  --config projects/price_tag_pipeline/configs/hq_ensemble.yaml \
  --output outputs/jsonl/video01.jsonl \
  --log-level INFO
```

## 4. Visual Debug Video

Render boxes, recognized fields, QR values, and OCR/QR read moments directly on
top of the original video:

```bash
python projects/price_tag_pipeline/scripts/visualize_predictions.py \
  --video data/raw/videos/video01.mp4 \
  --pred outputs/jsonl/video01.jsonl \
  --audit outputs/audit.jsonl \
  --out outputs/vis/video01_annotated.mp4 \
  --linger-frames 45
```

If there is no audit file, omit `--audit`; boxes and final fields are still drawn.

The visualizer is intentionally simple:

- bounding boxes show final unique tags;
- labels show prices/product/weight/QR barcode;
- the top-left panel shows the exact frame where OCR or QR decoded something.

## 5. Hackathon CSV

```bash
python projects/price_tag_pipeline/scripts/export_hack_csv.py \
  --inputs outputs/jsonl \
  --out-csv submission/hack_submission.csv \
  --video-ext .mp4
```

The CSV columns match the PDF task:

- `filename`, product fields, prices, barcode/SKU/date/code/color/symbols;
- bbox coordinates and `frame_timestamp` in milliseconds;
- QR payload fields such as `price1_qr`, `action_price_qr`, `action_code_qr`.

Policy:

- if a field is recognized, it is filled;
- if a VLM/QR explicitly returns `нет`, the CSV keeps `нет`;
- if the pipeline cannot recognize a field, the cell is empty.

## 6. Gradio UI

```bash
python projects/price_tag_pipeline/scripts/gradio_app.py \
  --config projects/price_tag_pipeline/configs/zeroshot_nolabel.yaml \
  --outputs-dir outputs/demo \
  --host 0.0.0.0 \
  --port 7860
```

Open:

```text
http://localhost:7860
```

Upload a video. The app returns:

- annotated MP4;
- hackathon CSV;
- short run summary.

## 7. Reading Quality Quickly

Use the annotated video first. Watch for:

- missed price tags: detector threshold too high or prompt labels too narrow;
- many duplicate boxes: tracking/dedup thresholds too loose;
- boxes on products instead of tags: open-vocab labels need tuning;
- correct QR but wrong visible text: trust QR fields, tune OCR parser;
- text read only on sharp stop frames: increase `top_k_crops_per_track`.

Fast threshold knobs live in:

```text
projects/price_tag_pipeline/configs/zeroshot_nolabel.yaml
```

Important knobs:

- `detector.conf`
- `detector.open_vocab_labels`
- `ocr.min_sharpness`
- `ocr.min_crop_area_px`
- `aggregation.min_final_confidence`
- `aggregation.dedup_iou_threshold`
