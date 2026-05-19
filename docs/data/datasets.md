# External datasets — expanding a 5-video set

The organizers provide **5 videos with a ground-truth CSV** (~63 frames,
~274 tag rows — see [`layout.md`](./layout.md)). That is far too small to
train a detector or an OCR head from scratch. Our quality lever is therefore
**external open datasets + open-source-model auto-labeling + synthetic
Lenta-style tags**, then fine-tuning on strong camera-matched augmentations
— no human-in-the-loop labeling. This file is the curated shortlist of that
external data, ordered by expected lift for *our* task (price-tag
**detection** + **structured field OCR** on Russian retail tags).

> Nothing here is auto-downloaded. `scripts/fetch_external_datasets.py` prints
> the plan and only fetches a target you explicitly pass `--download`, with
> your own API keys. Most sets need a license click / email — respect it.

> **This is the short list.** The full landscape study (15+ datasets scored
> for the robot-shelf domain — products, SKU, price tags, facings/counting,
> OOS, planogram — plus a recommended modular architecture and a web-found
> extensions appendix incl. **RusTitW**, **BarBeR**, Cyrillic-OCR synthetics)
> is in [`datasets-research.md`](../internal/datasets-research.md). Read that for the
> *why* and for the **killer-feature** (facings/OOS) dataset backing; this
> file is the actionable shortlist for the core task.

## Tier 0 — biggest lever: synthetic Lenta-style tags (no download)

We know the exact tag schema (29 CSV columns: prices, loyalty/card price,
discount %, weight, QR block, wholesale tiers, Cyrillic product names). The
2024 IEEE study *Multi-Class Price Tag Detection in Supermarket Shelves* lifted
detector mAP **73.1 → 93.9%** purely by template-based synthetic price tags +
duplicate-and-shift augmentation. With only 5 videos this is the highest-ROI
path: render synthetic RU tags from templates, composite onto shelf
backgrounds, pair with the heavy Albumentations pipeline already in the repo.
Tracked as future work in [`../strategy.md`](../strategy.md).

## Tier 1 — detector pretraining / domain warm-up

| Dataset | Size | Why it helps | Access |
|---|---|---|---|
| **SKU-110K** (Goldman et al., CVPR'19) | 11,762 imgs, densely packed shelves | Best match for *small, dense, look-alike* retail boxes; pretrain the detector backbone before fine-tuning on our 5 videos. Ultralytics ships `SKU-110K.yaml` for one-line use. | [eg4000/SKU110K_CVPR19](https://github.com/eg4000/SKU110K_CVPR19) · [Ultralytics docs](https://docs.ultralytics.com/datasets/detect/sku-110k/). Academic/non-commercial; email request, but Ultralytics yaml auto-mirrors. |
| **Grocery Store Dataset** (Klasson et al.) | 5,125 imgs, 81 classes + iconic label images | Shelf/product context + clean "iconic" label crops for an OCR/classification aux task. | [marcusklasson/GroceryStoreDataset](https://github.com/marcusklasson/GroceryStoreDataset). Research use. |

## Tier 2 — on-domain price-tag detection (small, fine-tune/val)

Roboflow Universe — small but exactly price tags; good as extra fine-tuning /
sanity-val data. Need `ROBOFLOW_API_KEY` + `pip install roboflow`.

| Dataset | ~Imgs | Note |
|---|---|---|
| [CUHK `price-tag-mpq14`](https://universe.roboflow.com/cuhk-00cw9/price-tag-mpq14) | 81 | price-tag + digit boxes |
| [Andra `price-tag`](https://universe.roboflow.com/andra-acalfoaie/price-tag) | 50 | plain price-tag boxes |
| [`shelf-tag-label-assist`](https://universe.roboflow.com/shelf-tag-scanning/shelf-tag-label-assist) | 127 | ships a pretrained model + API |
| [SDP `price-labelling`](https://universe.roboflow.com/sdp-project/price-labelling) | — | priced-item classes |

Licenses vary per dataset (mostly CC BY 4.0); check each project page.

## Tier 3 — OCR / digits / barcode side

| Resource | Why | Access |
|---|---|---|
| [Kaggle: OCR receipts text detection](https://www.kaggle.com/datasets/trainingdatapro/ocr-receipts-text-detection) | Printed RU/EN prices & digits — weak for tag layout, useful for price/number CER. | `kaggle` CLI + `kaggle.json` |
| **BarBeR** (Barcode Benchmark Repository) | Our CSV has `qr_code_barcode` + `price{1..4}_qr`; harden QR/1D decoding (pairs with `src/price_tag_pipeline/recognition/` — `qr.py` / `barcode.py`). | search "BarBeR barcode benchmark" |
| [FutureBee product-label / price-tag OCR](https://www.futurebeeai.com/dataset/product-label-ocr-image-data-sets) | Largest RU product-front/price-tag OCR corpus found. | Commercial, request-based (note: likely paid) |

## Recommended order of work

1. **Synthetic RU tags** (Tier 0) — independent of any download, biggest lift.
2. **SKU-110K pretrain** → fine-tune detector on our 5 videos + Tier 2.
3. Hold one organizer video out (`data/splits/`) as the only *real* val signal;
   external sets are train-only (domain shift — they are not Lenta tags).

## Current external processing pipeline

The reproducible intake/conversion script is
[`projects/price_tag_pipeline/scripts/prepare_external_datasets.py`](../../projects/price_tag_pipeline/scripts/prepare_external_datasets.py).
It reads raw downloads from `../dataset_research/` by default, or from
`DATASET_RESEARCH_ROOT` when that environment variable is set.

It builds task-specific research datasets instead of mixing incompatible labels:

```bash
python projects/price_tag_pipeline/scripts/prepare_external_datasets.py status
python projects/price_tag_pipeline/scripts/prepare_external_datasets.py build-available
```

Current processed-output notes are in
[`external-dataset-processing-report.md`](../internal/external-dataset-processing-report.md).

Sources: [SKU-110K](https://github.com/eg4000/SKU110K_CVPR19),
[Ultralytics SKU-110K](https://docs.ultralytics.com/datasets/detect/sku-110k/),
[Roboflow retail universe](https://universe.roboflow.com/browse/retail),
[Grocery Store Dataset](https://github.com/marcusklasson/GroceryStoreDataset),
[Multi-Class Price Tag Detection (IEEE)](https://ieeexplore.ieee.org/iel7/10278502/10278578/10279709.pdf),
[Grocery label review (MDPI)](https://www.mdpi.com/2076-3417/13/5/2871).
