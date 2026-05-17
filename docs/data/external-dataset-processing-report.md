# External Dataset Processing Report

Date: 2026-05-17.

Goal: download the requested external retail datasets, keep raw data auditable,
and split usable data by model/training purpose instead of mixing incompatible
labels into one detector dataset.

## Download status

| Dataset | Status | Local path | Notes |
|---|---|---|---|
| OpenFoodFacts price-tag-detection | Already present and processed | `dataset_research/openfoodfacts_price_tag_detection/raw/` | Hugging Face parquet train/val are present. |
| HITL Supermarket Shelves | Downloaded and extracted | `dataset_research/hitl_supermarket_shelves/raw/` | Downloaded from Kaggle API endpoint as `hitl_supermarket_shelves.zip`; extracted Supervisely JSON annotations. |
| SKU-110K | Downloaded and extracted | `dataset_research/sku110k/raw/SKU110K_fixed/` | Downloaded from the official S3 URL used by Ultralytics docs. Kept as product/object data only. |
| SOVAR Roboflow price-tag-detection v2 | Manually exported and processed | `dataset_research/sovar_price_tag_detection/raw/price tag detection.v2i.yolov8/` | YOLOv8 export is present. Broad and filtered-clean tiers were built. |

## Generated task datasets

Generated under `dataset_research/task_datasets/`.

| Task dataset | Images/crops | Boxes | Purpose |
|---|---:|---:|---|
| `price_tag_detector/openfoodfacts_lenta_strict_candidates_v1` | 435 | 1,925 | Cleaner shelf-context candidate set for price-tag detector training. |
| `price_tag_detector/openfoodfacts_price_tag_broad_v1` | 593 | 3,598 | Broader price-tag pretraining set; includes useful but less Lenta-like cases. |
| `ocr_tag_crops/openfoodfacts_full_tag_crops_v1` | 5,522 | n/a | Full-tag crop candidates for OCR smoke tests and generic readability experiments. |
| `price_tag_detector/sovar_roboflow_price_tag_broad_v1` | 994 | 2,665 | Full SOVAR YOLOv8 export collapsed to `price_tag`; keep mainly for audit/pretraining experiments. |
| `price_tag_detector/sovar_roboflow_price_tag_clean_v1` | 398 | 922 | Filtered SOVAR tier after removing many black/gray redaction boxes. |
| `price_tag_detector/hitl_supermarket_shelves_price_tag_broad_v1` | 41 | 1,762 | Broad detector training from HITL `Price` boxes. |
| `product_context/hitl_supermarket_shelves_products_v1` | 45 | 9,924 | Product/context boxes for hard-negative/product auxiliary work; not price tags. |
| `object_pretrain/sku110k_object_v1` | 11,743 | 1,730,996 | Dense product/object detector pretraining only; never price-tag positives. |
| `composites/external_price_tag_clean_train_v1` | 874 | 4,609 | Main clean external price-tag train mix: OFF strict + SOVAR clean + HITL Price. |
| `composites/external_price_tag_broad_train_v1` | 1,467 | 8,207 | Broader external price-tag train mix with OFF broad and filtered SOVAR. |
| `composites/external_object_pretrain_v1` | 11,788 | 1,740,920 | SKU-110K object boxes plus HITL products for dense object/product warm-up. |

## Processing rules

### OpenFoodFacts

Input is the previous `clean_candidates_v0` YOLO-style subset, but it is now
split more aggressively:

- `lenta_strict_candidate`: max box area <= 0.18, median box area <= 0.055,
  median aspect in 1.25..6.0, no edge-cut boxes, and a shelf-context proxy
  (`>=2` tags or one small tag).
- `price_tag_broad`: remaining usable OpenFoodFacts candidate images.
- `ocr_full_tag_crop_candidate`: per-box crops, excluding very large boxes.

Why: the old automatic candidate set was good as a first pass, but still mixed
macro tags, electronic shelf labels, loose boxes, and shelf-like frames. The
new split preserves broad signal while giving a cleaner first training tier.

### HITL Supermarket Shelves

Input is Supervisely JSON:

- `classTitle == "Price"` -> YOLO class `0 price_tag`;
- `classTitle == "Product"` -> separate YOLO class `0 product`;
- product boxes are never merged into price-tag positives.

Why: HITL is valuable precisely because products and price labels are both
annotated. Keeping products separate lets us build hard-negative/product-context
experiments without poisoning the price-tag detector.

### SKU-110K

SKU-110K is product/object data only. When the archive is available, it should
be converted to `object_pretrain/sku110k_object_v1` with class `object`.

It must not be merged as `price_tag`. The value is dense shelf/object warm-up,
occlusion learning, and future facing/product experiments.

### SOVAR Roboflow

The pipeline expects a YOLO export with `data.yaml`. When the export is present,
it collapses only full tag/label/price-like classes to `price_tag` and excludes
digit/product classes. Roboflow augmentations must be inspected because the
public v2 page reports stretch-to-640 and heavy augmentation.

Additional handling added after visual QA:

- `sovar_roboflow_price_tag_broad_v1` preserves the mapped export for audit.
- `sovar_roboflow_price_tag_clean_v1` filters out many black/gray redaction
  boxes using per-crop darkness, brightness, texture, and chroma checks.
- `analysis/box_quality_report.csv` records per-box quality metrics and whether
  each source box was kept.

## Visual QA notes

I inspected generated contact sheets:

- `hitl_supermarket_shelves_price_tag_broad_v1/qa_contact_sheet.jpg`: boxes are
  generally centered on shelf price tags and very useful for broad detector
  training. Some images have tiny labels, angled shelves, non-Lenta fixtures,
  and dense price rows; keep as train/pretrain only.
- `hitl_supermarket_shelves_products_v1/qa_contact_sheet.jpg`: product boxes are
  dense and correctly separate products from price labels. This is good context
  or hard-negative material, not price-tag positive data.
- `openfoodfacts_lenta_strict_candidates_v1/qa_contact_sheet.jpg`: much cleaner
  than the old automatic candidate bucket. It still contains foreign/electronic
  shelf labels and some loose/non-Lenta templates, so it is a candidate tier
  pending manual QA, not trusted validation.
- `sovar_roboflow_price_tag_broad_v1/qa_contact_sheet.jpg`: useful shelves and
  price labels, but many labels are black/gray redacted. Broad SOVAR should not
  be used as-is for clean detector training.
- `sovar_roboflow_price_tag_clean_v1/qa_contact_sheet.jpg`: substantially
  cleaner; remaining images can still contain redacted background regions, so
  use it as candidate train data with lower trust than HITL/OFF strict.
- `sku110k_object_v1/qa_contact_sheet.jpg`: annotations correctly cover dense
  product objects/facings. This is a good object-pretraining source and must
  remain separate from price-tag detector labels.

## Repro commands

```bash
python projects/price_tag_pipeline/scripts/prepare_external_datasets.py status
python projects/price_tag_pipeline/scripts/prepare_external_datasets.py download-runbook
python projects/price_tag_pipeline/scripts/prepare_external_datasets.py build-available
```

Each detector output has:

- `dataset.yaml`;
- `train_images.txt`;
- `val_images.txt` for smoke checks only;
- `manifest.csv`;
- `qa_contact_sheet.jpg`.

Use real held-out Lenta video for final validation.
