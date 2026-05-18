# Shelf Analytics Layer

This is the optional product/facing layer for visible facings, SKU grouping,
and price-tag association. It is intentionally separate from the graded
price-tag CSV path.

Current stable path:

```text
video -> price_tag detector -> QR/barcode/OCR -> FinalTag -> 29-column CSV
```

Shelf analytics path:

```text
keyframe/video segment
  -> product_facing detections
  -> optional SKU/retrieval metadata
  -> grouping
  -> recognized price tags from the existing pipeline
  -> price_tag_for_group relations
  -> shelf_state.json
```

## Code

The pure-Python core lives in:

```text
projects/price_tag_pipeline/src/price_tag_pipeline/shelf_analytics/
```

- `schema.py` defines `ProductFacing`, `PriceTagObservation`, `ShelfGroup`,
  `TagProductRelation`, and `ShelfState`.
- `grouping.py` groups product facings by explicit review hints, confident
  `sku_id`, or embeddings. Detection classes stay generic.
- `matcher.py` links recognized price tags to product groups using same-row,
  horizontal-overlap, below-distance, center-distance, and text-similarity
  scores.
- `pipeline.py` assembles the final `ShelfState` and adapts existing
  `FinalTag` objects into `PriceTagObservation`.
- `product_detector.py` adapts the existing generic `Detection` contract into
  `ProductFacing`, so a separate product detector can be plugged in without
  touching the current price-tag detector.

## Data Strategy

No new store visit is needed for the first prototype:

1. Use the existing labeled Lenta videos for price tags.
2. Use the existing unlabeled videos for product-facing auto-label review.
3. Use SKU-110K/HITL product-context as generic product/facing pretraining.
4. Manually review only a small set of keyframes for product boxes, grouping,
   and price-tag relations.

The missing data is not generic product detection. The missing data is:

- target-store SKU reference images/crops;
- relation labels: which price tag belongs to which product group;
- hard negatives: shifted tags, promo tags, similar packages, reflections.

Keep `shelf_state.json` physically separate from the 29-column CSV so shelf
analytics can evolve without harming the scored price-tag pipeline.
