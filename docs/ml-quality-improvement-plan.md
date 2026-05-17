# ML quality improvement plan

Date: 2026-05-17. Scope: improve the graded price-tag CSV pipeline by fixing
the full chain:

```text
detect -> crop/rectify -> QR/OCR -> parse -> track vote -> dedup -> CSV
```

Main conclusion: do not optimize "the model" as one black box. If detection,
crop quality, QR/OCR, parsing, or aggregation fails, the final CSV is bad even
when another stage looks strong in isolation.

This document is the execution plan. It combines the current local findings,
the stage-by-stage critique, and the external/synthetic dataset strategy from
[`data/datasets.md`](./data/datasets.md) and
[`data/datasets-research.md`](./data/datasets-research.md).

## 1. Current state

Known data:

- Labeled organizer data: **5 videos**, **274 price-tag boxes**, about **63
  annotated frames**.
- Unlabeled organizer data: **3 videos**.
- Video resolution: 3840x2160, about 20 FPS.
- Current train-ready path exists: Lenta CSV -> extracted frames -> YOLO labels
  via `prepare_data.py`.
- Current production configs default to OpenFoodFacts'
  `hf://openfoodfacts/price-tag-detection/weights/best.pt`; a later local
  Lenta fine-tuned checkpoint should replace it only after held-out validation.
- Current practical detector training path is **Ultralytics YOLO**. RF-DETR is
  documented but not yet implemented as a working training/inference path.
- SAHI/TTA/WBF utilities exist, but SAHI is not yet wired into the main
  `detector.py` / `pipeline.py` inference path.
- Current pipeline already has crop buffering and top-K crop selection per
  track, but the crop score is still mostly sharpness/area/detection confidence.

Observed local failure mode from the current inference notes:

- YOLO-World zero-shot finds weak/noisy candidates and many false positives.
- Classical PaddleOCR returned empty text on small/rotated GT crops in the
  smoke checks.
- Full zero-shot smoke output can be empty because detections do not become
  useful parsed observations.

Working hypothesis for current quality issues:

| Failure source | Expected share | Why |
|---|---:|---|
| Detector misses / false positives | 40% | Zero-shot/open-vocab is not specific enough for Lenta tags. |
| Bad crops | 25% | Bboxes can be tight, small, tilted, blurred, or partly reflective. |
| Generic OCR over full tag | 20% | Full-tag OCR mixes product name, prices, QR, barcode, dates, and service text. |
| Parser / field mapping | 10% | Price and barcode formats are irregular but mostly deterministic. |
| Tracking / dedup / voting | 5% | Already reasonably designed, but source priorities and dedup keys need work. |

## 2. North-star metric

The final hackathon metric is tag-level CSV quality: a GT-matched tag passes
when enough substantive fields are correct. Barcode is especially important
because it is a primary matching key.

For engineering, optimize intermediate gates first. The final CSV score is too
late and too opaque to tell which stage broke.

## 3. Stage-by-stage metrics

Every meaningful experiment should output this table. If a metric regresses,
stop and inspect that stage before changing downstream code.

| Stage | Metric | How to measure | P0 target | Stretch target |
|---|---|---|---:|---:|
| Detector | `detector_recall@0.5` | YOLO val on held-out organizer video | >= 0.90 | >= 0.95 |
| Detector | `detector_precision@0.5` | YOLO val on held-out organizer video | >= 0.50 | >= 0.75 |
| Detector | `avg_detections_per_frame` | inference JSON/debug counters | bounded; inspect visually | stable per aisle |
| Detector | `false_positive_types` | manual QA on annotated videos | top FP types known | packages/shelf rails rare |
| Crop | `crop_gate_pass_rate` | accepted crops / detections | >= 0.60 | >= 0.80 |
| Crop | `crop_readable_rate` | manual QA or OCR non-empty proxy | >= 0.60 | >= 0.85 |
| QR | `qr_decode_rate_on_gt_crops` | QR extractor on GT boxes | establish baseline | >= 0.70 where QR exists |
| QR | `qr_decode_rate_on_pred_crops` | QR extractor on detector crops | >= 0.40 | close to GT-crop rate |
| OCR | `ocr_non_empty_rate` | non-empty OCR / OCR calls | >= 0.60 | >= 0.85 |
| OCR | `price_zone_ocr_rate` | any price candidate found | >= 0.70 | >= 0.90 |
| Parser | `price_parse_rate` | parsed price / readable tag | >= 0.70 | >= 0.90 |
| Parser | `barcode_parse_rate` | parsed barcode / readable tag | >= 0.60 | >= 0.85 |
| Parser | `product_name_non_empty_rate` | non-empty product name / readable tag | >= 0.50 | >= 0.80 |
| Aggregation | `dedup_duplicate_rate` | duplicate final rows / final rows | <= 0.15 | <= 0.05 |
| Aggregation | `track_field_fill_rate` | non-empty substantive fields / final row | >= 0.60 | >= 0.85 |
| Final CSV | `matched_tag_pass_rate` | `eval_hack_csv.py` where GT exists | improve each run | highest available |

Important interpretation:

- Low detector recall -> fix detector, SAHI, data, or thresholds.
- Good detector but low crop/QR/OCR rates -> fix crop/rectification/zone OCR.
- OCR non-empty but low parse rates -> fix parser and normalization.
- Good per-track fields but bad CSV -> fix source-priority voting and dedup.

## 4. Experiment loop

Use this loop for every detector/OCR change:

1. Run detector eval on a held-out organizer video.
2. Run pipeline with debug counters:
   `detections -> crops accepted -> QR reads -> OCR calls -> non-empty OCR ->
   parsed observations -> finalized tags`.
3. Render annotated videos.
4. Watch false positives and missed tags.
5. Compare the stage metric table.
6. Keep only changes that improve the target stage without hurting downstream
   metrics.

Never select a checkpoint by loss curve alone. With five videos, mAP can lie;
visual QA on real video is mandatory.

## 5. P0: make detection usable

Goal: replace zero-shot as the main path with a trained one-class price-tag
detector.

### P0.1 Train one-class detector

Use:

- model: `yolo11n.pt` for smoke, then `yolo11s.pt` or `yolo11m.pt`;
- class schema: one class, `price_tag`;
- `imgsz`: 1280 first, 1536 if GPU allows;
- inference `conf`: sweep `0.05`, `0.08`, `0.10`, `0.15`, `0.20`;
- objective: **high recall first**, then reduce false positives.

Commands are maintained in
[`runbooks/colab-private.md`](./runbooks/colab-private.md).

Do not train product/package classes into the detector that feeds OCR. Product
detection can be a separate future model, but the graded OCR pipeline needs
price-tag crops only.

### P0.2 Add hard negatives and active labels

The current 274 boxes are not enough. Use the 3 unlabeled videos:

1. Run the trained detector at low confidence.
2. Collect false positives: bottles, boxes, shelf rails, logos, product QR-like
   graphics, reflections.
3. Add frames with false positives to the training set as hard-negative images
   with either no label or corrected price-tag labels.
4. Add missed true tags as new positive labels.
5. Retrain and compare `detector_recall@0.5`, false-positive type counts, and
   annotated videos.

### P0.3 Tune for recall before OCR

Initial runtime config:

```yaml
detector:
  conf: 0.08
  iou: 0.60
  image_size: 1536
```

If false positives become too expensive, first improve hard negatives and crop
gates before raising confidence too aggressively. Missing a real tag is usually
worse than sending an extra candidate to QR/OCR.

### P0.4 Use heavy augmentation as a first-class training lever

With only 5 labeled videos and 274 boxes, heavy augmentation is mandatory. The
goal is to teach the detector the real robot conditions before we have enough
real labels: motion blur, small/far tags, glare, angle, compression, dark shelf
zones, partial occlusion, and scale changes.

Use the existing augmentation code:

- `training/augmentation.py` for Ultralytics built-in augmentation parameters;
- `training/augmentation_albu.py` for the heavier Albumentations profile;
- `--use-albu` in `train_detector_yolo.py` for serious detector runs.

Start with these augmentation families:

| Augmentation | Why it matters | Initial strength |
|---|---|---|
| Motion blur | robot moves along shelves | high, p around 0.30-0.40 |
| Defocus / Gaussian blur | camera focus and tiny text | medium |
| Brightness / contrast | aisle lighting changes | high |
| CLAHE | low contrast tags and shadows | low-medium |
| ISONoise / GaussNoise | low light / sensor noise | medium |
| Perspective | side viewing angles | medium |
| Random shadow / glare | shelf covers and reflections | medium |
| Hue/saturation jitter | preserve red/yellow/green tag colors but vary lighting | low-medium |
| Scale / translate | tags appear at many sizes/positions | high |
| Coarse dropout | shelf rail, product edge, hands, partial occlusion | medium |
| Horizontal flip | shelf direction symmetry | enabled |
| Vertical flip / 90-degree random rotate | breaks text orientation | disabled |

Important: "as much augmentation as possible" means **broad but controlled**.
If augmentation becomes unrealistic, validation on real held-out videos will
drop. The acceptance rule is simple: keep an augmentation only if real-video
recall or crop/QR/OCR downstream metrics improve.

Recommended sweeps:

```text
aug_light:  built-in Ultralytics only
aug_heavy:  --use-albu current heavy profile
aug_blur+:  heavier motion blur / defocus
aug_glare+: stronger brightness, shadow, flare
aug_scale+: stronger scale/perspective, imgsz 1536
```

Metrics to compare:

- `detector_recall@0.5` on held-out real video;
- `detector_precision@0.5`;
- false-positive types on annotated videos;
- `crop_gate_pass_rate`;
- `qr_decode_rate_on_pred_crops`;
- `ocr_non_empty_rate`.

Do not select augmentation by train mAP. Select by held-out real video and
downstream crop readability.

## 6. P0: crop quality and QR-first extraction

Goal: make detected tags readable before spending time on OCR model changes.

### P0.5 Increase crop padding

Start with:

```yaml
rectifier:
  padding_ratio: 0.20
```

Sweep `0.12`, `0.20`, `0.30`, `0.35`. Choose by `qr_decode_rate`,
`ocr_non_empty_rate`, and manual crop QA. Slightly more background is better
than cutting off QR, price, or product-name text.

### P0.6 Generate crop variants per detection

Current code keeps top-K crops across frames. It should also test several
variants of the same crop:

- padded 12%;
- padded 20%;
- padded 35%;
- CLAHE luminance;
- threshold/sharpened;
- 2x/3x upscale after rectification.

Do not OCR every variant blindly forever. Use them for QR and for selecting the
best OCR candidate, then cache the winner per track.

### P0.7 Make QR aggressive

QR is the highest-trust source for many fields. Improve `QRCodeExtractor` usage
so each selected crop tries:

- 1x, 2x, 3x scale;
- BGR/RGB original;
- grayscale;
- CLAHE;
- adaptive threshold;
- QR-region guesses if available.

If QR succeeds on any crop in a track, attach the decoded QR payload to the
track and reuse it for all later voting.

Source priority:

```text
QR field > visible barcode detector/parser > high-confidence OCR > VLM fallback > empty
```

QR-derived `barcode`, `price*_qr`, wholesale levels, `action_price_qr`, and
`action_code_qr` should beat conflicting OCR/VLM values.

## 7. P0/P1: layout-aware OCR and deterministic parser

Goal: stop running OCR as if a full price tag were one plain text document.

### P0.8 Split OCR by zones

After orientation normalization, create zones:

```text
full crop
├── top 40%       -> product_name candidates
├── middle/right  -> large prices
├── bottom 30%    -> barcode, id_sku, print_datetime, code
└── QR region     -> QR decoder, not OCR-first
```

These zones can begin as simple proportional crops. Later they can become
template-aware using [`hackathon/price-tag-guide.md`](./hackathon/price-tag-guide.md).

### P0.9 Build price OCR ensemble

For the price zone:

1. OCR original price zone.
2. OCR thresholded price zone.
3. OCR enlarged price zone.
4. Extract regex candidates.
5. Score candidates by plausibility.

Parser rules:

- normalize `129⁹⁹`, `129 99`, `129,99`, `129.99`, `129-99`, `I29.99`;
- keep prices inside plausible range;
- `price_card` is usually <= `price_default`, but do not enforce this as a
  hard rule because some mechanics can violate monotonicity;
- kopecks like `00`, `49`, `59`, `89`, `90`, `99` are more likely than random
  digits, but this is a soft prior.

### P0.10 Keep VLM as fallback, not the base path

Use deterministic extraction as the main path:

```text
trained YOLO -> QR -> PaddleOCR/zoned OCR -> deterministic parser
```

Use one selected VLM only when:

- QR failed;
- OCR confidence is low;
- product name / additional info is empty;
- the crop passed readability gates.

Do not spend time now on many VLM backends or VLM LoRA. VLM fine-tuning only
makes sense after we have stable, correctly cropped tag images and JSON targets.

## 8. P1: wire SAHI and improve small-tag recall

Goal: recover small/far price tags in 4K frames.

SAHI utilities already exist in `inference/sahi_adapter.py`, but the main
runtime path does not expose them yet.

Add config:

```yaml
sahi:
  enabled: true
  tile_size: 1024
  overlap: 0.20
  merge_iou: 0.50
```

Runtime modes:

```text
fast      -> full-frame YOLO
balanced  -> full-frame YOLO, no SAHI by default
hq        -> YOLO + SAHI tiled inference
```

Measure:

- `detector_recall@0.5` on held-out organizer video;
- `avg_detections_per_frame`;
- false-positive type counts;
- total runtime per video.

Only keep SAHI if it improves real-video recall enough to justify extra
candidate count and runtime.

## 9. P1: source-aware aggregation and stronger dedup

Current aggregation votes per field and dedups by IoU/time. Improve source
awareness and duplicate merging.

### P1.1 Source-priority voting

Introduce source weights:

| Source | Weight |
|---|---:|
| QR payload | 1.00 |
| visible barcode / barcode parser | 0.90 |
| high-confidence zoned OCR | 0.70 |
| selected VLM fallback | 0.50-0.70 |
| weak OCR regex | 0.30 |

If QR and OCR conflict on QR-owned fields, QR wins.

### P1.2 Dedup by identity, not only geometry

Robot motion means the same physical tag can appear at different pixel
coordinates after tracker loss. Add merge keys in this order:

1. exact QR barcode match;
2. exact visible barcode match;
3. exact `id_sku` match;
4. product-name similarity + price similarity;
5. crop visual embedding similarity;
6. bbox IoU + time window as a weak fallback.

Rule of thumb:

- barcode match -> merge almost always;
- product + price match -> merge likely;
- bbox-only match -> merge cautiously.

## 10. External and synthetic datasets

The additional datasets should be reflected in experiments, but they must not
pollute the target. No public dataset exactly matches Lenta robot video, so use
them module-by-module.

### Tier 0: synthetic Lenta-style tags

Priority: highest.

Why:

- We know the 29-column schema and tag layouts from the hackathon docs.
- Synthetic tags can generate exact bboxes and exact field text.
- They directly target price tags, unlike product shelf datasets.

Use for:

- detector positives;
- crop/rectifier QA;
- zone OCR pretraining/testing;
- parser edge cases.

Validation:

- never validate only on synthetic data;
- final detector choice still depends on held-out real organizer videos.

### Tier 1: SKU-110K / dense shelf data

Priority: useful but dangerous.

Use for:

- retail shelf visual warm-up;
- robustness to dense shelves and small objects.

Risk:

- SKU-110K is product/facing detection, not price-tag detection.
- If used naively, it can teach the model to detect packages, exactly the false
  positives we want to avoid.

Policy:

- use only as pretraining or background/hard-negative context;
- final fine-tune target remains one class: `price_tag`;
- validate on real organizer videos.

### Tier 2: Roboflow price-tag datasets

Priority: good for extra detector positives after label QA.

Examples listed in [`data/datasets.md`](./data/datasets.md):

- CUHK `price-tag-mpq14`;
- Andra `price-tag`;
- `shelf-tag-label-assist`;
- SDP `price-labelling`.

Policy:

- inspect labels manually;
- normalize to `price_tag`;
- remove product boxes unless intentionally used as hard negatives;
- keep licenses documented;
- train-only or sanity-val only, not the final real-video validation signal.

### Tier 3: OCR / barcode datasets

Use later, after crop quality works.

Useful directions:

- printed Cyrillic OCR data for product names;
- receipt/price OCR data for numeric parsing;
- BarBeR or similar barcode/QR benchmarks for decoder robustness.

Do not use these before we can read GT price-tag crops. If OCR is empty on GT
crops, fix crop/zone/decoder first.

## 11. Concrete backlog

### P0 - immediate quality lift

| Task | Output | Metric gate |
|---|---|---|
| Fine-tune YOLO one-class detector from the OpenFoodFacts baseline | local checkpoint + config override | recall@0.5 >= 0.90 on held-out real video |
| Add detector experiment notebook/runbook outputs | eval logs + annotated videos | every run comparable |
| Add hard-negative loop from unlabeled videos | extra labeled frames | FP types decrease |
| Run heavy augmentation sweeps | comparable YOLO runs | real-val recall/crop readability improve |
| Increase and sweep crop padding | config/profile update | QR/OCR rates improve |
| Add QR multi-scale/preprocess attempts | QR debug report | pred-crop QR rate moves toward GT-crop QR rate |
| Add pipeline debug counters | JSON/CSV metrics report | stage table available per run |
| Add simple OCR zones | zoned OCR outputs | price_parse_rate improves |
| Strengthen price/barcode parser | parser tests | parse edge cases pass |

### P1 - small-object and aggregation robustness

| Task | Output | Metric gate |
|---|---|---|
| Wire SAHI into hq detector path | config + implementation | recall gain > runtime/candidate cost |
| Add crop variants and crop readability score | crop candidate selector | OCR non-empty and QR rate improve |
| Add source-priority field voting | aggregator change | QR/barcode conflicts resolved correctly |
| Add identity-based dedup | dedup report | duplicate rate <= 0.15, then <= 0.05 |
| Add synthetic tag generator | synthetic YOLO + OCR crops | improves real-val detector/crop metrics |
| Add Roboflow price-tag import/normalization | external train set | real-val recall or FP profile improves |
| Add augmentation ablation report | table of augmentation profiles | keep only real-video winners |

### P2 - robustness and stretch

| Task | Output | Metric gate |
|---|---|---|
| Visual crop embeddings for dedup | embedding matcher | duplicate rate down without bad merges |
| One selected VLM fallback | fallback OCR profile | product/additional fields improve |
| SKU-110K warm-up experiment | pretrained detector run | real-val detector improves after fine-tune |
| Field status: recognized / absent / unknown | export semantics | fewer `"нет"` vs empty mistakes |
| Product/facing analytics as separate model | non-graded report | does not affect graded CSV |

## 12. What not to prioritize now

Avoid spending time on:

- RF-DETR before YOLO + SAHI + data loops are exhausted.
- VLM LoRA before stable crop/JSON pairs exist.
- Many VLM backends.
- Product service integration.
- Frontend polish.
- Complex async backend work.

The likely quality lift comes from:

```text
trained detector
+ hard negatives and external/synthetic data
+ SAHI for small tags
+ better crops
+ QR-first extraction
+ zoned OCR
+ deterministic parser
+ source-aware voting and dedup
```

## 13. Definition of done for the next milestone

The next milestone is complete when:

1. `balanced.yaml` uses the OpenFoodFacts detector or a better validated local
   fine-tuned detector, not YOLO-World zero-shot.
2. Every experiment produces the stage metric table.
3. Detector recall on held-out organizer video is at least 0.90 at IoU 0.5.
4. Annotated videos show most real tags found and false-positive classes known.
5. QR decode is attempted with multi-scale preprocessing.
6. OCR runs on zones, not only full crops.
7. Aggregation gives QR/barcode fields source priority.
8. Final CSV improves over zero-shot and does not produce large duplicate noise.
