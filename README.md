# lenta_tech_life_2026

End-to-end price-tag recognition for the Lenta Tech Life 2026 hackathon
(Russian retail chain; price-tag detection + structured field extraction
from robot-captured video).

## Documents

- `ANALYSIS.md` — analysis of the original scaffold (what worked, what didn't, why we rewrote).
- `STRATEGY.md` — model-selection and pipeline strategy (RF-DETR / YOLO26 / PaddleOCR-VL / BoT-SORT).
- `ML_BASELINE_COLAB.md` — GPU-only ML baseline runbook for Google Colab (train/infer/eval/export).
- `data/README.md` — exactly where data goes when it arrives.
- `projects/price_tag_pipeline/README.md` — pipeline usage and CLI reference.

## Pipeline at a glance

```
video.mp4
  └─► YOLO26 / RF-DETR detector (1280 input, BoT-SORT tracker)
        └─► Per-track top-K-sharpest crop buffer
              └─► PaddleOCR-VL 1.5 with JSON-schema prompt
                    └─► Per-field weighted voting (fuzzy buckets for prices)
                          └─► Cross-track IoU + content dedup
                                └─► outputs/{video}.jsonl
```

## Quick start

```bash
# 1. Install (Python 3.11 or 3.12 recommended; 3.14 is too new for some deps).
pip install -r projects/price_tag_pipeline/requirements.txt

# 2. Drop data under data/raw/ (see data/README.md).

# 3A. Zero-shot baseline (no labels, no training).
python projects/price_tag_pipeline/scripts/run_batch_inference.py \
    --videos-dir data/raw/videos \
    --config projects/price_tag_pipeline/configs/zeroshot_nolabel.yaml \
    --outputs-dir outputs/jsonl
python projects/price_tag_pipeline/scripts/export_hack_csv.py \
    --inputs outputs/jsonl --out-csv submission/hack_submission.csv

# 3B. Trainable path (if labels are available): prepare + split + train + infer.
python projects/price_tag_pipeline/scripts/prepare_data.py     --raw data/raw --processed data/processed
python projects/price_tag_pipeline/scripts/make_splits.py      --processed data/processed --out data/splits --n_splits 5 --emit-dataset-yaml-fold 0
python projects/price_tag_pipeline/scripts/train_detector_yolo.py --dataset data/processed/dataset.yaml --model yolo26l.pt --imgsz 1280 --batch 8 --epochs 200
python projects/price_tag_pipeline/scripts/run_inference.py    --video path/to/video.mp4 --config projects/price_tag_pipeline/configs/balanced.yaml --output outputs/video01.jsonl
```

## Tests

```bash
pip install -r projects/price_tag_pipeline/requirements/dev.txt
pytest projects/price_tag_pipeline/tests -v
```

## Status

- Pre-data scaffold is complete. Drop data into `data/raw/` and the pipeline above is
  copy-pasteable.
- RF-DETR detector backend and PaddleOCR-VL LoRA fine-tuning are scaffolded as
  stubs; both land once data is available (see STRATEGY.md §§2.1, 3.3).
