# External + synthetic detector training plan

Date: 2026-05-17. Scope: make the **price-tag detector** train on synthetic
Lenta-style tags plus the external datasets discussed in
[`datasets.md`](../data/datasets.md) and [`datasets-research.md`](./datasets-research.md).

This document is intentionally practical: what is ready today, what is missing,
how every dataset should enter the pipeline, and what the training runs should
look like.

## 1. Short answer: readiness today

The repo is now **partly ready for external detector pretraining**, but not yet
ready for the full synthetic + curriculum training system.

Ready now:

- Lenta organizer CSV/video ingestion works:
  `scripts/ingest_real_data.py -> scripts/prepare_data.py`.
- `prepare_data.py` can normalize **Lenta CSV**, **YOLO labels**, and **COCO
  JSON** into the internal YOLO layout.
- Video-level real-data splits exist under `data/splits/fold_{0..4}.json`.
- YOLO detector training exists in
  `projects/price_tag_pipeline/scripts/train_detector_yolo.py`.
- Built-in and heavy Albumentations augmentation hooks exist.
- Detector evaluation exists in `scripts/eval_detector.py`.
- `scripts/prepare_external_datasets.py` can build the currently available
  external research sets under `dataset_research/task_datasets/`, including
  OpenFoodFacts strict/broad price-tag tiers, HITL price/product splits,
  SOVAR broad/clean tiers, SKU-110K object pretraining, and composite train
  mixes. See `external-dataset-processing-report.md`.
- Runtime detection defaults to OpenFoodFacts'
  `hf://openfoodfacts/price-tag-detection/weights/best.pt` before local
  fine-tuning.

Not ready yet:

- No synthetic Lenta tag generator exists.
- No source-aware sampler beyond materialized composite train lists exists.
- External/synthetic data still must not be used for final model selection;
  validation should remain held-out real Lenta video.
- `train_detector_yolo.py` accepts one `dataset.yaml`; it does not know about
  per-source curriculum training.
- RF-DETR training is still a placeholder; the working training path is YOLO.

So the current code can train on **real Lenta**, **one normalized external
dataset**, or a **prebuilt external composite dataset**. The missing pieces are
synthetic Lenta rendering, explicit curriculum/source weighting, and validated
promotion of a local fine-tuned checkpoint over the OpenFoodFacts default.

## 2. Current data contract

All detector training should end in this canonical layout:

```text
data/processed/<dataset_name>/
├── frames/
│   └── <source_or_video_id>/
│       └── 000001.jpg
├── labels/
│   └── <source_or_video_id>/
│       └── 000001.txt
├── images -> frames              # symlink/junction/copy alias for Ultralytics
├── train_images.txt
├── val_images.txt
├── source_manifest.jsonl         # new: one row per image
└── dataset.yaml
```

Each label file is YOLO detect format:

```text
0 cx cy w h
```

For the graded detector, the class schema must stay:

```yaml
names:
  0: price_tag
```

Do **not** mix product boxes from SKU-110K/Grocery Store as `price_tag`. Product
datasets can be used for backbone warm-up or hard-negative/background training,
but not as positive price-tag labels.

## 3. Dataset-by-dataset integration plan

| Dataset | Detector value | Current support | Required work | Training role |
|---|---|---|---|---|
| **Synthetic Lenta-style tags** | Highest. Creates many target-domain price-tag boxes. | None. | Add renderer, compositor, manifest writer, QA previews. | Main positive training source. |
| **Real Lenta 5 videos** | Highest validation value; only true target domain. | Ready. | Keep one video held out; optionally add active labels. | Fine-tune + real validation only. |
| **OpenFoodFacts price-tag-detection** | Direct price-tag boxes, ~2.23k rows, production Open Prices origin. | Processed strict/broad tiers; runtime default detector uses its pretrained model. | Manual QA + held-out real Lenta validation before trusting as final. | External positive train source + default bootstrap model. |
| **Roboflow/SOVAR price-tag** | Direct price-tag boxes, small but on-domain. | Processed broad and filtered-clean tiers when export is present. | Continue manual QA; use clean tier first. | External positive train source. |
| **Other Roboflow price-tag sets**: CUHK, Andra, shelf-tag-label-assist, SDP | Direct but tiny/noisy. | Generic Roboflow helper can download one project. | Same as SOVAR; add registry entries and class remap. | Optional positive train; never validation. |
| **HITL Supermarket Shelves** | Product + Price boxes in shelf context, CC0. | Processed into separate Price and Product tiers. | Manual QA; keep Product separate. | External positive train + false-positive sanity. |
| **SKU-110K** | Dense shelf product pretraining, small-object shelf robustness. | Processed as object pretraining when archive is available. | Use only as separate warm-up/object dataset. | Backbone/head warm-up only; not mixed as price-tag positives. |
| **Grocery Store Dataset** | Shelf/product context, not price-tag boxes. | Download helper only. | Optional converter for background/hard negatives or product auxiliary tasks. | Optional hard-negative/background source. |
| **Kaggle receipts OCR** | Text/price OCR, not detector. | Download helper only. | OCR-specific converter; do not feed detector. | OCR pretraining/eval. |
| **BarBeR barcode benchmark** | Barcode/QR robustness, not detector. | None. | QR/barcode eval/pretraining pipeline. | QR/barcode module only. |
| **FutureBee OCR / commercial sets** | Potential OCR + tag images if license allows. | None. | Manual acquisition, license check, converter after sample inspection. | OCR or detector depending on labels. |

## 4. New scripts needed

### 4.1 `generate_synthetic_lenta_tags.py`

Purpose: create synthetic Lenta-style price tags and their exact bboxes.

Input:

- templates described by `docs/hackathon/price-tag-guide.md`;
- product-name pools from real CSV rows plus generated Cyrillic names;
- price/card/discount/date/barcode/QR-like values;
- background frames from real labeled/unlabeled videos and/or external shelf
  images.

Output:

```text
data/synthetic/lenta_tags_v1/
├── frames/synth_lenta_v1/*.jpg
├── labels/synth_lenta_v1/*.txt
├── previews/*.jpg
└── source_manifest.jsonl
```

Required rendering variants:

- 6x6 regular price tag;
- 6x6 promo tag with black discount circle;
- 6x12 horizontal tag;
- A5/A4 larger tags;
- threshold discount tag;
- QR present/absent;
- barcode graphic/text variants;
- white/yellow/red/green promo color variants;
- shelf-talker attached vs absent.

Required augmentations at generation time:

- perspective warp;
- scale from very small to medium;
- motion blur;
- defocus;
- glare/reflection overlays;
- JPEG compression;
- partial crop at image edge;
- occlusion by shelf rail/product rectangle;
- low-light and color-temperature shift.

Important: the synthetic renderer only needs to make tags visually useful for
**detection**. OCR-perfect text is not required for the detector, but text-like
layout must look realistic.

### 4.2 `fetch_external_datasets.py` expansion

Current helper covers `sku110k`, one generic `roboflow`, `kaggle-receipts`, and
`grocery-store`.

Add explicit targets:

```text
openfoodfacts-price-tags
hitl-supermarket-shelves
roboflow-sovar-price-tag
roboflow-cuhk-price-tag
roboflow-andra-price-tag
roboflow-shelf-tag-label-assist
```

Each target should download into:

```text
data/external/<dataset_id>/raw/
```

Do not silently download commercial/request-only sets. For FutureBee/Unidata,
the script should print instructions and expected local placement.

### 4.3 `convert_external_detector_dataset.py`

Purpose: normalize each external dataset into canonical one-class YOLO
`price_tag`.

Required adapters:

- `openfoodfacts_hf`: Hugging Face dataset with `image`, `width`, `height`,
  `objects.bbox`, `objects.category_name`.
- `roboflow_yolov8`: already YOLO-like; class remap required.
- `hitl_kaggle`: inspect downloaded format, then map only `Price`/price-tag
  boxes to class `0`.
- `empty_negative_images`: copy background/hard-negative images with empty
  `.txt` labels.

Output:

```text
data/processed/external_<dataset_id>/
├── frames/<dataset_id>/*.jpg
├── labels/<dataset_id>/*.txt
├── source_manifest.jsonl
└── dataset.yaml
```

Every converter must:

- clamp boxes to image bounds;
- drop zero-area boxes;
- write empty label files for hard-negative images;
- record original source path/license/split in `source_manifest.jsonl`;
- run `validate_dataset()`;
- render a `previews/` grid with boxes for quick QA.

### 4.4 `build_detector_mix.py`

Purpose: create a trainable dataset from many normalized sources.

Input example:

```yaml
name: detector_mix_v1
class_names: [price_tag]
sources:
  - id: real_lenta_train
    processed: data/processed/lenta_real
    include: train_only
    weight: 8.0
  - id: synthetic_lenta_v1
    processed: data/synthetic/lenta_tags_v1
    include: train_only
    weight: 4.0
  - id: openfoodfacts_price_tags
    processed: data/processed/external_openfoodfacts_price_tags
    include: train_only
    weight: 2.0
  - id: roboflow_sovar
    processed: data/processed/external_roboflow_sovar
    include: train_only
    weight: 1.0
  - id: hitl_shelves
    processed: data/processed/external_hitl_shelves
    include: train_only
    weight: 1.0
validation:
  real_lenta_fold: 0
```

Output:

```text
data/processed/detector_mix_v1/
├── frames/...
├── labels/...
├── train_images.txt
├── val_images.txt        # real Lenta held-out fold only
├── source_manifest.jsonl
└── dataset.yaml
```

The first implementation can oversample by repeating image paths in
`train_images.txt`. Later, replace with a custom sampler if needed.

Validation rule:

- `val_images.txt` must contain only held-out real Lenta frames.
- External/synthetic data must never be used to select the final checkpoint by
  validation metric.

## 5. Training curriculum

### Phase A: optional dense-shelf warm-up

Use SKU-110K as a separate warm-up run, because its labels are products, not
price tags.

```bash
yolo detect train \
  model=yolo11x.pt \
  data=SKU-110K.yaml \
  imgsz=1280 \
  epochs=30 \
  batch=8 \
  project=runs/lenta \
  name=sku110k_warmup
```

Result: a checkpoint with shelf/small-object bias. It is not a price-tag model
yet.

### Phase B: price-tag detector pretrain

Train on synthetic + OpenFoodFacts + Roboflow + HITL, with no real validation
selection except a held-out Lenta fold.

```bash
python projects/price_tag_pipeline/scripts/build_detector_mix.py \
  --config projects/price_tag_pipeline/configs/data/detector_mix_v1.yaml

python projects/price_tag_pipeline/scripts/train_detector_yolo.py \
  --dataset data/processed/detector_mix_v1/dataset.yaml \
  --model runs/lenta/sku110k_warmup/weights/best.pt \
  --epochs 100 \
  --imgsz 1280 \
  --batch 8 \
  --device 0 \
  --name detector_mix_v1_pretrain \
  --use-albu
```

If Phase A is skipped, start from
`hf://openfoodfacts/price-tag-detection/weights/best.pt` or another available
YOLO checkpoint.

### Phase C: real Lenta fine-tune

Fine-tune with real Lenta oversampled strongly and synthetic/external still
present at lower weight.

Suggested mix:

| Source | Approximate sampling weight |
|---|---:|
| Real Lenta train folds | 8 |
| Synthetic Lenta tags | 4 |
| OpenFoodFacts price tags | 2 |
| Roboflow price tags | 1 |
| HITL Price boxes | 1 |
| Hard negatives from unlabeled videos | 2 |

Command:

```bash
python projects/price_tag_pipeline/scripts/train_detector_yolo.py \
  --dataset data/processed/detector_mix_v1/dataset.yaml \
  --model runs/lenta/detector_mix_v1_pretrain/weights/best.pt \
  --epochs 80 \
  --imgsz 1536 \
  --batch 4 \
  --device 0 \
  --name detector_mix_v1_lenta_ft \
  --use-albu
```

### Phase D: active hard-negative loop

1. Run inference on the 3 unlabeled videos and on held-out real videos at low
   confidence.
2. Render detections.
3. Add false-positive frames as empty-label hard negatives.
4. Add missed price tags as new positive labels.
5. Rebuild the mix and retrain.

Repeat until false positives are mostly explainable and held-out real recall
stops improving.

## 6. Evaluation rules

Detector selection metrics:

- primary: recall on held-out real Lenta fold;
- secondary: mAP@0.5 and mAP@0.5:0.95 on held-out real Lenta fold;
- tertiary: visual QA on rendered videos;
- downstream: QR/OCR non-empty rate on predicted crops.

Do not trust external validation for the final decision. External sets have
different countries, camera positions, tag shapes, and label policies.

Required report per run:

```text
run_name
checkpoint
data_mix_config
real_val_fold
imgsz
augmentation_profile
mAP@0.5
mAP@0.5:0.95
precision
recall
avg_detections_per_frame
top false-positive types
top missed-tag types
qr_decode_rate_on_pred_crops
ocr_non_empty_rate_on_pred_crops
```

## 7. Important pitfalls

### Product datasets are not price-tag positives

SKU-110K and Grocery Store have many product boxes. If those boxes are mapped
to `price_tag`, the model will learn to detect products instead of tags. Use
them only as:

- separate warm-up training;
- background/hard-negative images with empty labels;
- future product/facing detector tasks.

### Empty labels are useful but dangerous

Hard-negative frames should have empty `.txt` labels only if we are confident
there are no visible target price tags in the frame. Otherwise we teach the
model to ignore true tags.

### Synthetic must be visually audited

Bad synthetic data can hurt. For every synthetic version, render at least 200
preview images with bboxes and reject the version if tags look too clean,
cartoonish, oversized, or unlike robot video.

### Hold-out must be real Lenta only

The final detector should be selected by held-out organizer video. External
validation is useful for debugging converters, not for deciding whether the
model is good for this task.

## 8. Minimal implementation order

1. Add `generate_synthetic_lenta_tags.py` with 6x6 regular/promo templates and
   preview rendering.
2. Add OpenFoodFacts converter; this is the strongest immediate external
   price-tag dataset.
3. Add Roboflow YOLO converter/class remapper for SOVAR and the other tiny
   Roboflow price-tag sets.
4. Add HITL converter, keeping only `Price` boxes.
5. Add `build_detector_mix.py` with source-aware oversampling and real-only
   validation.
6. Train `detector_mix_v1_pretrain`.
7. Fine-tune `detector_mix_v1_lenta_ft`.
8. Run active hard-negative mining on unlabeled videos.
9. Only after YOLO saturates, wire RF-DETR or another detector as a second
   model for comparison/ensemble.

## 9. Expected first usable commands

After the missing scripts are implemented, the intended workflow should be:

```bash
# 1. Stage real Lenta data.
python projects/price_tag_pipeline/scripts/ingest_real_data.py \
  --src real_data/dataset \
  --dst data/raw

python projects/price_tag_pipeline/scripts/prepare_data.py \
  --raw data/raw \
  --processed data/processed/lenta_real

python projects/price_tag_pipeline/scripts/make_splits.py \
  --processed data/processed/lenta_real \
  --out data/splits \
  --n_splits 5 \
  --emit-dataset-yaml-fold 0

# 2. Generate synthetic detector data.
python projects/price_tag_pipeline/scripts/generate_synthetic_lenta_tags.py \
  --backgrounds data/raw/unlabeled_videos \
  --out data/synthetic/lenta_tags_v1 \
  --n-images 20000 \
  --seed 42

# 3. Fetch and convert external price-tag sets.
python projects/price_tag_pipeline/scripts/fetch_external_datasets.py \
  --download openfoodfacts-price-tags

python projects/price_tag_pipeline/scripts/convert_external_detector_dataset.py \
  --adapter openfoodfacts_hf \
  --src data/external/openfoodfacts-price-tags/raw \
  --out data/processed/external_openfoodfacts_price_tags

python projects/price_tag_pipeline/scripts/fetch_external_datasets.py \
  --download roboflow-sovar-price-tag

python projects/price_tag_pipeline/scripts/convert_external_detector_dataset.py \
  --adapter roboflow_yolov8 \
  --src data/external/roboflow-sovar-price-tag/raw \
  --out data/processed/external_roboflow_sovar

# 4. Build one mixed detector dataset with real-only validation.
python projects/price_tag_pipeline/scripts/build_detector_mix.py \
  --config projects/price_tag_pipeline/configs/data/detector_mix_v1.yaml

# 5. Train and evaluate.
python projects/price_tag_pipeline/scripts/train_detector_yolo.py \
  --dataset data/processed/detector_mix_v1/dataset.yaml \
  --model hf://openfoodfacts/price-tag-detection/weights/best.pt \
  --epochs 120 \
  --imgsz 1536 \
  --batch 4 \
  --device 0 \
  --name detector_mix_v1 \
  --use-albu

python projects/price_tag_pipeline/scripts/eval_detector.py \
  --weights runs/lenta/detector_mix_v1/weights/best.pt \
  --dataset data/processed/detector_mix_v1/dataset.yaml \
  --imgsz 1536 \
  --device 0
```

## 10. Definition of done

This training system is ready when:

- every source has a deterministic converter;
- every converted dataset has a preview grid and validation report;
- every training image has a `source_manifest.jsonl` row;
- `detector_mix_v1/dataset.yaml` is reproducible from config;
- validation contains only held-out real Lenta frames;
- a trained checkpoint improves held-out real recall over real-only training;
- rendered videos show fewer missed tags without exploding false positives;
- downstream crop/QR/OCR metrics improve, not just detector mAP.
