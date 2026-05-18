# Shelf Analytics Layer

Optional product/facing pipeline for visible facings, SKU grouping, and
price-tag association. This layer is a business/demo extension and must stay
physically separate from the graded 29-column price-tag CSV pipeline.

## Why This Exists

The current core pipeline answers:

```text
Where are price tags, and what fields can we read from them?
```

Shelf analytics answers a different question:

```text
Which visible product group is this price tag for, and how many facings are on
the shelf?
```

These are related but not the same task. Mixing product classes into the
price-tag detector would make OCR worse and could corrupt the scored CSV path.
The product/facing path therefore runs beside the price-tag pipeline, not
inside it.

## Current Stable Path

```text
video
  -> price_tag detector
  -> tracker
  -> rectified crops
  -> QR -> barcode -> OCR
  -> FinalTag
  -> 29-column CSV
```

Owned by:

- `projects/price_tag_pipeline/src/price_tag_pipeline/pipeline.py`
- `projects/price_tag_pipeline/src/price_tag_pipeline/recognition/`
- `projects/price_tag_pipeline/src/price_tag_pipeline/submission.py`

Do not route shelf analytics output into the 29-column CSV.

## Additional Shelf Analytics Path

```text
video/keyframes
  -> product_facing detector
  -> product crops
  -> optional SKU retrieval / embeddings
  -> grouping into same-SKU or unknown clusters
  -> recognized price tags from the existing pipeline
  -> price_tag_for_group matcher
  -> shelf_state.json
```

Implemented pure-Python seam:

```text
projects/price_tag_pipeline/src/price_tag_pipeline/shelf_analytics/
```

Files:

- `schema.py` — `ProductFacing`, `PriceTagObservation`, `ShelfGroup`,
  `TagProductRelation`, `ShelfState`.
- `product_detector.py` — adapts existing generic `Detection` into
  `ProductFacing`.
- `grouping.py` — groups product facings by review hints, confident `sku_id`,
  or embedding similarity.
- `matcher.py` — links recognized price tags to product groups using geometry
  and optional text similarity.
- `pipeline.py` — assembles `ShelfState` and adapts existing `FinalTag` into
  `PriceTagObservation`.

## Output Contract

The report is intentionally JSON-like and independent from hackathon CSV:

```json
{
  "image_id": "frame_000123",
  "product_facings": [
    {
      "id": "product_42_0",
      "bbox_xyxy": [100, 100, 160, 260],
      "sku_id": "milk_1l",
      "sku_confidence": 0.91,
      "row_id": "row_2"
    }
  ],
  "groups": [
    {
      "id": "sku_milk_1l_001",
      "member_facing_ids": ["product_42_0", "product_43_0"],
      "facing_count": 2,
      "sku_id": "milk_1l",
      "status": "known_sku"
    }
  ],
  "price_tags": [
    {
      "id": "tag_9",
      "bbox_xyxy": [95, 270, 230, 305],
      "price": 99.9,
      "product_name": "Молоко пастеризованное 1л"
    }
  ],
  "relations": [
    {
      "type": "price_tag_for_group",
      "price_tag_id": "tag_9",
      "product_group_id": "sku_milk_1l_001",
      "score": 0.74,
      "status": "ok"
    }
  ]
}
```

## Model Stack

### 1. Product/facing detector

Purpose: find each visible product facing, not identify SKU as a detector
class.

Recommended model:

- YOLO family via the existing Ultralytics runtime.
- Start small for smoke: `yolo11n.pt` / `yolo11s.pt`.
- Use higher `imgsz` (`1280+`) because shelf objects are dense and small.

Pretraining / warm-up data:

| Source | Role | Local status |
|---|---|---|
| SKU-110K | Dense retail object/facing pretraining. Teaches packed shelves and many adjacent boxes. | Prepared under `../dataset_research/task_datasets/object_pretrain/sku110k_object_v1/`. |
| HITL Supermarket Shelves product context | Small shelf product boxes; useful as sanity check and product-context data. | Prepared under `../dataset_research/task_datasets/product_context/hitl_supermarket_shelves_products_v1/`. |
| Existing unlabeled Lenta videos | Target-domain auto-label + human review. | `/Users/cute/Lenta_Tech/Данные/Unlabeled/`. |
| Own reviewed Lenta keyframes | Final target fine-tune and validation. | Needed; can be small for MVP. |

Important boundary:

- Keep the detector class generic: `product_facing`.
- Do not train `heineken_05_can`, `milk_1l`, etc. as YOLO classes.
- SKU identity belongs to retrieval, not detection.

### 2. Price-tag detector and OCR

Purpose: provide reliable price-tag boxes and recognized fields.

This already exists in the core pipeline:

- OpenFoodFacts price-tag detector base:
  `hf://openfoodfacts/price-tag-detection/weights/best.pt`
- Lenta fine-tuning path:
  `projects/price_tag_pipeline/scripts/train_detector_yolo.py`
- Recognition chain:
  `QR -> barcode -> OCR`

Training / support data:

| Source | Role | Local status |
|---|---|---|
| Lenta organizer videos | Target-domain price-tag boxes and CSV fields. | `data/processed/`, 5 videos, about 274 tag rows. |
| OpenFoodFacts price-tag data | Broad price-tag detector pretraining/fine-tune support. | Prepared in `../dataset_research/task_datasets/price_tag_detector/`. |
| SOVAR Roboflow price-tag data | Additional price-tag shapes and hard negatives. | Prepared in `../dataset_research/task_datasets/price_tag_detector/`. |
| Synthetic Lenta-style tags | Best future lift for OCR and detector augmentation. | Planned; no external download needed. |

The shelf analytics layer consumes `FinalTag` outputs through
`price_tag_from_final_tag()`.

### 3. SKU retrieval / identity

Purpose: decide whether product crops are the same SKU and, when possible,
match them to a known catalog item.

Recommended MVP path:

```text
product bbox
  -> crop
  -> image encoder embedding
  -> nearest reference SKU
  -> threshold
  -> sku_id or unknown
```

Good starting encoders:

- DINOv2 — strong visual similarity on product crops.
- CLIP / SigLIP — useful when text/product-name semantics help.
- Later: contrastive fine-tune on own SKU pairs and hard negatives.

Reference catalog data:

```json
{
  "sku_id": "milk_brand_x_1l",
  "barcode": "460...",
  "display_name": "Молоко Brand X 1л",
  "reference_images": [
    "refs/milk_brand_x_1l/front_01.jpg",
    "refs/milk_brand_x_1l/shelf_crop_01.jpg"
  ]
}
```

Useful pretraining / benchmark sources:

| Source | Role |
|---|---|
| SHAPE / SHARD | Shelf-row, SKU, and planogram-style reasoning reference. |
| RP2K / RPC | Fine-grained retail product recognition/retrieval. |
| GroZi-3.2K / GroZi-120 | Product image to shelf image retrieval reference. |
| Own Lenta shelf crops | Required for target-domain SKU quality. |

Production rule:

- New SKU should be added by updating the reference catalog, not by retraining
  a giant detector classifier.

### 4. Grouping

Purpose: merge multiple visible facings into one product group.

Current implementation:

```text
ProductFacing[] -> group_facings() -> ShelfGroup[]
```

Grouping priority:

1. `group_hint` from annotation/review UI.
2. Confident `sku_id`.
3. Embedding similarity within the same shelf row.
4. One unknown group per facing if no reliable signal exists.

This lets the MVP work even before exact SKU recognition exists:

- known SKU groups when catalog/retrieval is confident;
- unknown clusters when products look visually identical;
- conservative fallbacks when the model is unsure.

### 5. Shelf row detection

Purpose: constrain product/tag matching to the same shelf row.

MVP options:

- Rule-based row estimation from product/tag y-coordinates.
- Optional `shelf_edge` detector later.
- Manual `row_id` in reviewed keyframes for validation.

Pretraining ideas:

- SHARD / SHAPE / Shelf Management papers as architecture references.
- Own keyframes are still needed because shelf geometry depends on camera
  height, robot route, and store layout.

### 6. Price-tag to product-group matching

Purpose: assign each recognized price tag to the group it describes.

Current implementation:

```text
ShelfGroup[] + PriceTagObservation[] -> match_price_tags() -> relations[]
```

Score components:

- horizontal overlap;
- price tag below product group;
- x-center distance;
- same `row_id`;
- optional text similarity between SKU/display name and OCR `product_name`.

Current policy:

- high score and clear winner -> `status = ok`;
- low score -> `status = ambiguous`;
- close competing candidates -> `status = ambiguous`;
- keep top candidates in the output for review/debug.

This is intentionally rule-based first. It creates debuggable labels and can
later be replaced or assisted by a learned relation model.

## Data Needed

No new store visit is required for the first prototype. The first prototype
can use:

- existing 5 labeled Lenta videos for price tags;
- existing 3 unlabeled Lenta videos for product auto-label review;
- SKU-110K and HITL product-context for generic product detector warm-up;
- existing price-tag datasets for price-tag detector robustness.

What is still missing:

| Need | Minimum useful amount | Why |
|---|---:|---|
| Product-facing boxes on Lenta keyframes | 50-150 keyframes | Target-domain product detector validation/fine-tune. |
| Same-SKU group labels | 50-150 keyframes | Evaluate grouping and tune embedding threshold. |
| `price_tag_for_group` relations | 50-150 keyframes | Tune matcher and measure association error. |
| SKU reference catalog | 5-20 crops per SKU | Exact SKU retrieval and unknown threshold. |
| Hard negatives | As discovered | Similar packs, shifted tags, promo tags, reflections. |

For MVP, do not label every frame. Extract diverse keyframes, auto-label them,
then manually correct only the useful cases.

## Training / Build Plan

### Phase 0 — no new labels

1. Use existing price-tag pipeline to produce `FinalTag`.
2. Run a generic product detector or open-vocabulary baseline on keyframes.
3. Feed product detections + `FinalTag` into `build_shelf_state_from_pipeline_outputs()`.
4. Inspect `shelf_state.json` visually/manually.

Expected result: facings and rough price associations, many unknown SKUs.

### Phase 1 — small review set

1. Select 50-100 diverse keyframes from existing videos.
2. Correct product boxes.
3. Mark obvious same-SKU groups.
4. Mark price-tag relations where unambiguous.
5. Tune grouping and matcher thresholds.

Expected result: stable demo on known Lenta footage.

### Phase 2 — product detector fine-tune

1. Pretrain/warm up on SKU-110K/HITL product-context.
2. Fine-tune on reviewed Lenta keyframes.
3. Validate on held-out target keyframes.

Metrics:

- product/facing mAP;
- small-object recall;
- over-merge/split visual QA.

### Phase 3 — SKU retrieval

1. Build initial reference catalog from shelf crops and packshots.
2. Compute DINOv2/CLIP/SigLIP embeddings.
3. Tune top-1/top-3 and unknown threshold.
4. Add hard negatives for similar SKU.

Metrics:

- top-1 accuracy;
- top-3 accuracy;
- unknown detection precision/recall;
- confusion matrix for similar packages.

### Phase 4 — association validation

1. Use reviewed relation labels.
2. Tune row/overlap/distance/text weights.
3. Measure `price_tag_for_group` correctness.

Business metric:

```text
% product groups with correct matched price tag
```

## Evaluation

Do not report a single blended number. Track each layer:

| Layer | Metric |
|---|---|
| Product detector | mAP@50, recall on small/occluded facings. |
| Grouping | same-SKU grouping precision/recall, split/merge error. |
| SKU retrieval | top-1, top-3, unknown threshold quality. |
| Price OCR | price exact match, barcode recall, product-name fuzzy score. |
| Matching | correct price-tag relation rate, ambiguous rate, false match rate. |
| End-to-end | correct SKU + correct price + correct facing count. |

## Failure Modes

- Similar SKU with different flavor/volume grouped together.
- Price tag shifted under a neighboring product.
- Promo tag or shelf-talker mistaken for regular price tag.
- Glass reflections / robot motion blur create false products.
- One RGB frame cannot infer stock depth behind the front row.
- Academic SKU datasets do not reflect current Lenta assortment.

## Deployment Boundary

The additional pipeline should export:

```text
outputs/shelf_state/<video_or_frame>.json
```

The scored pipeline should keep exporting:

```text
submission/hack_submission.csv
```

Do not merge these outputs until there is an explicit product requirement. The
safe integration point is the review UI: show CSV results and shelf-state
results side by side, with separate download buttons.

## Current Code Example

```python
from price_tag_pipeline.shelf_analytics import build_shelf_state_from_pipeline_outputs

shelf_state = build_shelf_state_from_pipeline_outputs(
    image_id="frame_000123",
    product_detections=product_detections,
    price_tag_finals=final_tags,
)

payload = shelf_state.to_dict()
```

With enrichment:

```python
from price_tag_pipeline.shelf_analytics import ProductFacing, PriceTagObservation, build_shelf_state

state = build_shelf_state(
    image_id="frame_000123",
    product_facings=[
        ProductFacing(
            id="p1",
            bbox_xyxy=(100, 100, 160, 260),
            row_id="row_2",
            sku_id="milk_1l",
            sku_confidence=0.91,
            display_name="Молоко 1л",
        )
    ],
    price_tags=[
        PriceTagObservation(
            id="tag_9",
            bbox_xyxy=(95, 270, 230, 305),
            price=99.9,
            product_name="Молоко пастеризованное 1л",
            row_id="row_2",
        )
    ],
)
```

## Summary

The architecture is intentionally modular:

```text
price-tag CSV path: stable, scored, do not disturb
shelf analytics path: optional, business-facing, outputs shelf_state.json
```

Use public data for generic visual pretraining, but rely on a small reviewed
Lenta keyframe set for the two things public datasets cannot provide:

- exact target SKU/reference catalog;
- correct price-tag to product-group relations.
