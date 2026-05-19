# Experiments

The detector strategy is fixed: **we fine-tune the OpenFoodFacts price-tag
detector** (`hf://openfoodfacts/price-tag-detection/weights/best.pt`, a
YOLO11x) on our Lenta split. We tune *this* architecture from *these* weights
— we do not train a detector from scratch and we do not swap the backbone.

This directory pins experiment definitions as **declarative manifests**
(knobs + the exact command), not as a parallel runner. Execution stays in
`scripts/train_detector_yolo.py`, which already defaults to the base
checkpoint above. Keeping manifests separate from the runner means an
experiment is reviewable as one small file and reproducible by one command.

## Canonical experiment

- [`finetune_openfoodfacts.yaml`](./finetune_openfoodfacts.yaml) — the base
  checkpoint, dataset, hyper-parameters, run-naming, and the promotion gate.

## Loop

1. Prepare data + emit a fold:
   `scripts/prepare_data.py` → `scripts/make_splits.py --emit-dataset-yaml-fold N`
2. Fine-tune from the OFF base:
   `scripts/train_detector_yolo.py --dataset data/processed/dataset.yaml ...`
   (`--model` defaults to the OFF checkpoint; override only to compare.)
3. Evaluate vs the OFF baseline: `scripts/eval_detector.py --weights ...`
4. **Promote only if it wins**: set `detector.model_path` in
   `configs/balanced.yaml` (and `fast`/`hq`) to the new `best.pt`. Until a
   local checkpoint beats the baseline, inference stays on the OFF weights.

## Boundaries

- The recognition side (QR → barcode → smart OCR) is a *separate* concern
  with its own seam — see [`docs/recognition-pipeline.md`](../../../docs/recognition-pipeline.md).
  Detector experiments do not touch it.
- Full rationale, augmentations that worked, and the camera profile:
  [`docs/detector-finetuning-report.md`](../../../docs/internal/detector-finetuning-report.md)
  and [`docs/data/external-detector-training-plan.md`](../../../docs/internal/external-detector-training-plan.md).