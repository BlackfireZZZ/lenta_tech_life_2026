# Strategy — Price-Tag Recognition Pipeline (Lenta Tech Life 2026)

> Plan first, code second. All model choices below are verified against May 2026 SOTA via web search; sources at the end. Where I've made assumptions, they are marked **[ASSUMPTION]** and listed in §10.

> **⚠️ Update (2026-05-17). The assumptions in §10 are now resolved** against the
> official task and the organizers' chat. Read [`hackathon/task.md`](./hackathon/task.md)
> (formal spec) and [`hackathon/briefing.md`](./hackathon/briefing.md) (metric
> deep-dive + strategy) **first** — they override guesses in this document.
> The biggest deltas: (1) the output schema is **29 CSV columns / 30 fields**
> (18 tag + 11 QR), not the 9-field schema in §3.2; (2) the metric is *"≥80% of
> substantive fields correct on each GT-matched tag"* with **barcode as the
> primary matching key** — not the equal-weight Levenshtein target in §0;
> (3) **cloud APIs are banned at inference** and lightweight / `rknn int8` is
> rewarded — so the rent-an-A100 plan applies to *training only*, inference
> must be fully local. See the resolved §10 for the per-assumption mapping.

---

## 0. Optimization target

> "Model quality first. Inference speed secondary."

End-to-end score = **per-tag field accuracy** on a held-out video set. The single metric we optimize:

```
E2E = mean over tags of: (correct_regular_price) AND (correct_loyalty_price) AND (Levenshtein(product_name) ≤ τ)
                       AND (correct_weight_value AND correct_weight_unit)
                       AND (correct_price_per_unit, ±1 kopeck)
```

Plus standard:
- Detection: mAP@[.5:.95] (small/medium/large split — we will mostly land in small)
- OCR field-wise: CER, WER (per field)
- Final-report: precision/recall on (video, tag) pairs after cross-track dedup

We will tune for E2E. Detection mAP and field CER are sub-goals that feed it.

---

## 1. End-to-end pipeline (target architecture)

```
video.mp4
  └─► [Frame iterator + FPS read]                      ← read real FPS, not hardcoded 30
        └─► [RF-DETR / YOLO26 detector]               ← P0
              └─► [BoT-SORT tracker with CMC]         ← P1 (CMC = camera-motion compensation, fits moving robot)
                    └─► [Best-K-sharpest-frame picker per track]   ← P1
                          └─► [Perspective rectifier (4-pt homography)]  ← P1
                                └─► [PaddleOCR-VL 1.5 (0.9B) VLM
                                         with JSON-schema prompt]   ← P0
                                      └─► [Field validator + parser]
                                            └─► [Track-level multi-field voter]
                                                  └─► [Cross-track dedup
                                                        (IoU + content)]
                                                        └─► report.json / report.csv
```

Two-model fallback ladder for OCR/extraction so we don't bet everything on one path:
1. **Primary:** PaddleOCR-VL 1.5 (0.9B) with a fixed JSON schema prompt → fine-tuned on our train split if results allow.
2. **Fallback A:** Classical PaddleOCR (detection + recognition with Russian model) + rule-based parser (the rewritten parser).
3. **Fallback B:** Qwen3-VL-4B-Instruct fine-tuned on our annotated tags via LoRA (only if PaddleOCR-VL underperforms on our specific layouts).

Run **at least two paths** at inference and ensemble (majority vote per field) — costs latency but we have headroom.

---

## 2. Detection

### 2.1 Model choice

| Candidate | Why it's a contender | Verdict |
|---|---|---|
| **RF-DETR (Roboflow, ICLR 2026)** | DINOv2 backbone, SOTA on COCO **and** RF100-VL (domain transfer benchmark — the closest thing to "will this generalize to price tags"), first real-time model >60 AP@0.5:0.95, designed for fine-tuning. | **Primary.** Use RF-DETR-Base for training. RF-DETR-Large/2XL for final inference if compute allows. |
| **YOLO26 (Ultralytics, Oct 2025)** | NMS-free end-to-end, MuSGD optimizer + ProgLoss with STAL (specifically targeting small-object detection), 43% faster CPU. Easy training. | **Secondary baseline + ensemble partner.** Train YOLO26-l. Compare against RF-DETR; ensemble via Weighted Box Fusion (WBF) if both look strong. |
| YOLO12 | Attention-centric. Reported training instability, memory hungry. | Skip. |
| YOLO11 | Proven, stable, on Ultralytics native support. | Keep as a sanity check / fallback if RF-DETR/YOLO26 misbehave on our data. |
| Co-DETR / DINO-DETR | Strong on COCO, but slower convergence and trickier to fine-tune. | Skip unless we have spare compute and time. |

**Why a transformer detector here:** price tags are repetitive, small, and visually similar across the image (one shelf has 20+). Attention helps with global context; DINOv2 pretraining gives RF-DETR robustness that COCO-pretrained YOLOs lack on out-of-domain shelf imagery.

### 2.2 Input resolution and tiling

Price tags are **small** objects. We need either:
- **High input resolution** (1280 or 1600 short side for RF-DETR-Base), or
- **SAHI tiling** at inference (and optionally at training): 800×800 tiles with 20% overlap.

Plan:
1. Train at 1280 short side with multi-scale jitter [960, 1280, 1600].
2. At inference, run two modes: (a) full-frame 1280, (b) SAHI tiles 800×800/20%. Merge with WBF.

### 2.3 Augmentation (Albumentations, applied to both train and val→train cycle):

Calibrated to robot-driving-past-shelf conditions:

- **MotionBlur** (kernel 3–25 px, p=0.35) — robot pans cause blur; need strong invariance.
- **Defocus / GaussianBlur** (p=0.10)
- **ISONoise / GaussNoise** (p=0.20) — store lighting is often low-light.
- **RandomBrightnessContrast** (p=0.40) — supermarket lighting varies aisle-to-aisle.
- **CLAHE** (p=0.10) — match what the rectifier does.
- **RandomShadow + RandomSunFlare** (p=0.10) — glare/reflections on plastic tag covers.
- **HueSaturationValue** (p=0.20, low hue range) — preserve red/yellow promo colors.
- **Perspective** (scale 0.02–0.05, p=0.30) — viewing angle.
- **CoarseDropout** (max_holes=4, hole_size 4–16 px, p=0.15) — simulates partial occlusion.
- **Mosaic** off after the first 70% of epochs, max prob 0.3 (hurts small-object recall above that).
- **MixUp** disabled — confuses small-object regression heads.
- **HorizontalFlip** OK; **VerticalFlip off** (would invert price text orientation).

### 2.4 Training schedule

- RF-DETR-Base: 60 epochs, batch 8 (4070 Ti at 1280), AdamW, lr 1e-4 backbone / 5e-4 head, cosine decay, warmup 1000 iters.
- YOLO26-l: 200 epochs, batch 16 at 960 or batch 8 at 1280, default SGD with MuSGD on.
- EMA on for both.
- Mixed precision (bf16 if available on A100/H100, fp16 on 4070 Ti).
- Save: best mAP@0.5:0.95 + last; track lasts 5 epochs of EMA.

### 2.5 Validation strategy (CRITICAL)

**Video-level GroupKFold**, never frame-level. Plus:
- Stratify by `(store_id, aisle_id)` if metadata exists. **[ASSUMPTION]** — if not, stratify by store only or skip stratification.
- 5 folds for final model selection.
- Hold out one entire video for end-to-end evaluation; it never appears in train or val.

```python
# Pseudocode
folds = StratifiedGroupKFold(n_splits=5)
for train_idx, val_idx in folds.split(frames, y=store_id_per_frame, groups=video_id_per_frame):
    ...
```

Frames from the same video share lighting, store layout, and many tags — frame-level splits leak both visual style and *some of the same physical tags* across train/val. Video-level is non-negotiable.

---

## 3. OCR + structured extraction

> **⚠️ Prerequisite read: [`hackathon/price-tag-guide.md`](./hackathon/price-tag-guide.md).**
> It is the field-layout bible distilled from the organizers' decks — where
> every CSV field physically sits on each tag type/mechanic, and the rules
> that change parsing: `discount_amount` is `-N%` when the ruble discount
> < 100 ₽ else `-N₽` (so the same field carries both encodings); on 6×12 and
> threshold layouts the barcode is **text** (`ШК:`/`Ш:`), not a graphical EAN;
> А5 and МНЦ have **no QR** (write `"нет"` for all 11 QR fields); a
> shelf-talker is part of the **same logical tag** and its scale number maps
> to `additional_info`; `color` may come from the linked talker, not the
> body. The §3.2 prompt and any parser must be built against that guide, not
> the simplified 9-field sketch below.

### 3.1 Why a VLM, not classical OCR

A Lenta price tag is **graphically structured**: regular-price region, loyalty-card-price region, product-name block, weight block, price-per-unit small text. Classical OCR returns a soup of strings; a parser then has to recover which string is which field. That parser is brittle on Russian (declensions, abbreviations) and on graphical layouts (which price is which?).

A VLM with a JSON-schema prompt reads the tag and emits structured output directly. The current SOTA open VLM for OCR-grade text + structure is **PaddleOCR-VL 1.5 (0.9B)**, released early 2026:
- NaViT-style dynamic-resolution visual encoder + ERNIE-4.5-0.3B LM
- 94.5% on OmniDocBench v1.5 — beats Qwen2.5-VL-72B, GPT-4o, Gemini 2.5 Pro
- 109 languages including Russian
- Native structured Markdown / JSON output
- Fine-tunable with LoRA on consumer GPUs

### 3.2 Prompt skeleton (per crop)

```
You are reading a Russian supermarket price tag.
Return ONLY a JSON object with exactly these keys:
{
  "regular_price": "<RR.KK or null>",
  "loyalty_price": "<RR.KK or null>",
  "product_name": "<string in Russian or null>",
  "weight_value": "<number or null>",
  "weight_unit": "<'кг'|'г'|'л'|'мл'|'шт'|null>",
  "price_per_unit_value": "<number or null>",
  "price_per_unit_unit": "<string or null>",
  "promo_flag": "<true|false>",
  "currency": "RUB"
}
Rules:
- Prices in roubles with 2 kopeck digits.
- Use null for fields you cannot read with confidence.
- Loyalty/card price is usually larger/highlighted; regular price is crossed out or smaller.
- Do not invent. If unsure, output null.
```

### 3.3 Fine-tuning plan (Stage 3 work)

1. Annotate **≥1000 tag crops** with the JSON schema above using whatever ground truth the organizers give us. **[ASSUMPTION]** — if the GT only has prices, start there and add fields incrementally.
2. LoRA fine-tune PaddleOCR-VL 1.5 on (crop, JSON) pairs. r=16, alpha=32, target attention modules. 3–5 epochs, lr 5e-5.
3. Validate per-field CER and structural validity (does it return parseable JSON? schema-conforming?).
4. If JSON schema-violation rate >5%, switch to **constrained decoding** (grammar-constrained generation via Outlines / SGLang).

### 3.4 Classical OCR fallback

For frames where the VLM returns nulls or invalid JSON:
- PaddleOCR 3.x (`from paddleocr import PaddleOCR; PaddleOCR(lang='ru', use_textline_orientation=True)`) — the actual current API. The scaffold's API call is broken.
- Hand off detected text spans to the **rewritten parser** (handles superscript kopecks, thousand-separator spaces, multi-line major/minor digits, fuzzy currency).
- Color/region heuristic to assign price → regular vs loyalty:
  - Yellow/red dominant region → loyalty (promo) price.
  - White background, smaller/strikethrough → regular price.

### 3.5 Russian-specific concerns

- **Cyrillic OCR.** Confirmed: PaddleOCR-VL 1.5 supports Russian out of the box. Classical PaddleOCR has a Russian recognition model. TrOCR's open-source Russian variants are weaker than PaddleOCR's; skip.
- **Superscript kopecks (`129⁹⁹`).** VLM handles trivially; classical OCR may split the line — parser must reassemble.
- **Product names with abbreviations / brand names.** VLM > classical for this. Don't gate by dictionary — that destroys recall on brand names.
- **Weight encoding.** "1кг", "1 кг", "1,5 л", "500 г", "500г". All must normalize to (value, unit) where unit ∈ {`кг`, `г`, `л`, `мл`, `шт`}. Parser normalizes `г`→g, `кг`→kg, etc.

---

## 4. Tracking and deduplication

### 4.1 Tracker choice

| Candidate | Why | Verdict |
|---|---|---|
| **BoT-SORT** with CMC (camera motion compensation) | Robot is *moving*; objects are stationary. CMC explicitly models global motion → far fewer ID switches. | **Primary.** |
| ByteTrack | No CMC; will be confused by camera pans. Simpler. | Fallback / fast profile. |
| Deep OC-SORT | Re-ID head adds robustness to occlusion. Heavier. | Use only if BoT-SORT shows ID-switch issues. |

Ship `bytetrack.yaml` and `botsort.yaml` locally (under `projects/price_tag_pipeline/configs/trackers/`) so we tune `track_buffer`, `match_thresh`, `proximity_thresh`, `appearance_thresh`, and CMC method per profile.

### 4.2 Per-track frame selection

Currently the scaffold runs OCR every N frames per track. Better:
1. While a track is live, buffer up to T=15 crops with their sharpness, area, and detection confidence.
2. When the track is finalized (TTL expired or video ends), pick the **top-K=5 crops** by `0.6·sharpness + 0.4·area·det_conf`.
3. Run OCR on those K crops.
4. Vote per field (not per whole-record): each field gets its own majority vote weighted by `ocr_field_confidence × det_conf × sharpness`.

This decouples field-level correctness — a frame with great product_name but corrupted price still contributes good product_name.

### 4.3 Cross-track deduplication (post-aggregation)

Track-ID re-emission is real. After aggregation, run a dedup pass:
- Build a graph of finalized predictions in a 5-second sliding window.
- Edge if: `IoU(bboxᵢ, bboxⱼ) > 0.4` AND (`regular_priceᵢ == regular_priceⱼ` OR `Levenshtein(nameᵢ, nameⱼ) < 0.2 · len`).
- Merge connected components; pick the highest-confidence representative.

This protects against ID-switch double counting.

---

## 5. Data preparation

### 5.1 Annotation format

User says YOLO format for now, swappable. Plan:
- `prepare_data.py`: reads source annotations, validates them (no NaNs, no out-of-frame boxes, no zero-area boxes, class IDs match expected, every image referenced exists). Logs a integrity report.
- Outputs a YOLO-format dataset YAML plus a `splits.json` capturing fold membership at the **video** level.
- Tolerant of variations: COCO ↔ YOLO converter functions ready, so when the organizers reveal the real format we just swap a reader.

### 5.2 Integrity checks (must fail loudly)

- Every image referenced in labels exists and opens.
- Every label file's boxes are inside `[0, 1] × [0, 1]` (YOLO normalized).
- Box width/height > 0.
- Per-video frame count matches expectation if metadata is provided.
- Class distribution logged; any class with <50 examples warned about.
- Aspect-ratio and area histograms logged for sanity.

### 5.3 Video-level split

```python
groups = video_id_for_each_image
strata = store_id_for_each_image  # [ASSUMPTION] if metadata absent, fall back to a constant
splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
```

Manifests live under `data/splits/fold_{0..4}.json`, version-controlled.

---

## 6. Metrics

| Metric | Purpose | Tool |
|---|---|---|
| mAP@0.5:0.95 (size-stratified) | Detection quality | pycocotools / torchmetrics |
| Field CER (per field separately) | OCR/extraction quality | jiwer or local impl |
| Field Exact-Match (per field) | The thing graders care about | local impl |
| End-to-end record accuracy | The submission target | local impl |
| Cross-track dedup precision/recall | Sanity for double-counting | local impl |
| Latency profile (frames/sec, OCR calls/sec) | Tiebreaker if quality ties | local impl |

All metrics dumped to a JSON report and logged to W&B per epoch / per inference run.

---

## 7. Logging and reproducibility

- **Weights & Biases.** Project: `lenta-2026`. Two run types: `detector-{backbone}-{date}` and `e2e-eval-{checkpoint}-{date}`. Log dataset version (git SHA of `splits.json`), config dump, training curves, sample predictions every 5 epochs (10-image grid with bounding boxes), per-fold val mAP.
- Fixed seeds: `seed=42` in numpy, torch, ultralytics, and the sampler.
- `requirements.txt` → split into `requirements/base.txt` and `requirements/train.txt` and `requirements/eval.txt`. Pin everything we touch.
- Pre-commit hook: black + ruff + a tiny "no committed PyTorch checkpoints" guard.

---

## 8. Compute plan

| Job | Hardware | ETA |
|---|---|---|
| Data prep + integrity checks + 5-fold split manifest | local CPU | <30 min |
| YOLO26-l training, 5 folds at 960 input | local 4070 Ti, 12 GB | ~6h/fold |
| RF-DETR-Base training, 5 folds at 1280 input | rented A100 80GB or 4070 Ti at batch 4 | ~10h/fold on A100, ~24h on 4070 Ti |
| PaddleOCR-VL 1.5 LoRA fine-tuning | rented A100 (bf16) | ~3h per 1000 samples × 3 epochs |
| Full inference + report on held-out videos | local 4070 Ti | tens of minutes per video |

For RF-DETR + VLM LoRA: rent **1× A100 80GB** for ~24h. Otherwise local 4070 Ti can do YOLO26 + classical OCR pipeline.

---

## 9. Module-by-module rewrite plan (Stage 3)

| Module | Action |
|---|---|
| `config.py` | Extend with `TrainingConfig`, `EvalConfig`, `ReportConfig`, `AugmentationConfig`. Keep the dataclass pattern. |
| `types.py` | New `ParsedTag` with all fields (regular/loyalty/name/weight/ppu/currency). New `FieldVote` for per-field aggregation. |
| `detector.py` | Two implementations behind `BaseDetector`: `YOLOTrackerDetector` (Ultralytics, used for YOLO11/26) and `RFDETRDetector` (Roboflow SDK). FPS read from container. |
| `rectifier.py` | New `PerspectiveRectifier` that estimates a 4-pt homography from a corner-keypoint head OR falls back to padded axis-aligned crop. Keeps color (RGB), CLAHE only on the luminance channel. |
| `ocr.py` | New `PaddleVLMEngine` (calls PaddleOCR-VL 1.5 with the JSON prompt), keep `PaddleOCREngine` rewritten for 3.x API, drop noop default. |
| `parser.py` | Rewrite around the new schema. Multi-format price regex, superscript-kopeck handling, weight/unit normalizer. |
| `aggregator.py` | Multi-field voting; fuzzy keying for prices; per-track top-K-sharpest crop buffer; cross-track dedup pass. |
| `pipeline.py` | Orchestrate the new flow with the buffered crop selection and post-aggregation dedup. |
| `quality.py` | Add a "best crop in window" selector. |
| `metrics/` | NEW package. `detection.py`, `ocr.py`, `e2e.py`. |
| `training/` | NEW package. `train_detector.py`, `train_vlm_lora.py`, `dataset.py` (YOLO/COCO readers + Albumentations transforms). |
| `eval/` | NEW package. `eval_detector.py`, `eval_e2e.py`, `make_report.py`. |
| `scripts/` | NEW: `prepare_data.py`, `train_detector.py`, `eval_detector.py`, `train_vlm.py`, `run_inference.py` (rename of the current runner), `make_report.py`, `smoke_test.py`. |
| `configs/` | Split into `runtime/` (fast/balanced/hq), `training/` (per detector backbone), `eval/`. |
| `tests/` | Add smoke tests: parser end-to-end, aggregator multi-field, rectifier no-crash on edge boxes, detector dummy stream, eval runs on synthetic data. |

---

## 10. Assumptions — RESOLVED against the official task (2026-05-17)

The seven original assumptions are no longer open. Source of truth:
[`hackathon/task.md`](./hackathon/task.md) and
[`hackathon/briefing.md`](./hackathon/briefing.md). Mapping below; "Δ" = the
strategic change this forces.

| # | Original assumption | Resolved answer | Δ for this strategy |
|---|---|---|---|
| 1 | Annotation format = YOLO | **Lenta hackathon CSV** (`filename,product_name,price_default,price_card,…,frame_timestamp,x_min,y_min,x_max,y_max` + QR fields). `prepare_data.py` already ingests it; `frame_timestamp` is **ms**, not a frame index. | None for architecture; GT bboxes are noisy (briefing §6.1) — **build our own detection, don't reproduce GT boxes**. |
| 2 | 9-field output schema | **29 CSV columns** = 18 tag fields + 11 QR fields (task.md §3). `price_default`/`price_card`/`price_discount`, `barcode`, `id_sku`, `print_datetime`, `code`, `color`, `special_symbols`, plus 11 `*_qr` fields. | §3.2's 9-field VLM prompt is **insufficient** — extend it to the full schema; add `color` classification (white/yellow/green/red) and display-type (`к/л/ш`). |
| 3 | No per-video metadata | Confirmed — **none provided**. Store/aisle stratification is impossible. | Keep video-level GroupKFold *without* stratification (§2.5/§5.3 fallback path is the live one). |
| 4 | Rent A100 for training | Allowed for **training only**. **Cloud APIs / external online services are banned at inference** (task.md §10). Lightweight + `rknn int8` is explicitly rewarded. | Train heavy (RF-DETR / VLM LoRA) off-box if needed, but the **shipped inference pipeline must run fully local**. Add a lightweight edge profile (YOLO-n/s) for the on-robot scenario the organizers are weighing (briefing §8.2). |
| 5 | Metric weights fields equally | **Two-stage**: (A) match row→GT by **barcode** (primary key) else `frame_timestamp+bbox` with tolerances; (B) tag "recognized" iff ≥**80%** of *substantive* fields correct. Technical fields (`filename`, `frame_timestamp`, bbox) are **not scored**. | **Barcode recognition is now P0** — it is the matching key, not just a field. Duplicates actively hurt the score → cross-track dedup is critical. Plan QR as **11 separately-counted fields** (worst case). |
| 6 | Inference is per-video | Confirmed — **one CSV, one row per unique tag**. Submit **one** timestamp per tag (the best-recognition frame); time tolerance covers it. | Matches current pipeline + `export_hack_csv.py`. Keep the top-K-sharpest "best frame" selection (§4.2). |
| 7 | Pretrained weights allowed | **Yes**, any open-license model deployable locally. Manual labeling allowed for *training* only, never at inference, and must be disclosed in the README. | Proceed with DINOv2/COCO/PaddleOCR-VL pretrained checkpoints. Document every labeled/external dataset in the README (mandatory, task.md §7). |

**Net effect on the plan:** the architecture in §1 is still right, but (a) the
extraction schema must grow to 29 columns incl. `color`/`special_symbols`,
(b) **barcode + dedup move to P0**, (c) inference is **local-only** with an
optional lightweight edge profile, (d) "≥80% of fields" + barcode-keyed
matching replaces the equal-weight metric in §0. §11 below is re-ordered
accordingly in the runbooks; the original ordering is kept here for history.

---

## 11. Recommended starting order (Stage 3)

Once you confirm or override the above:

1. `prepare_data.py` + video-level split manifest (independent of detector choice).
2. `train_detector.py` with **YOLO26-l** first (it trains on the 4070 Ti without renting compute — get a baseline detection mAP within 24h).
3. Detection eval + cross-fold mAP report.
4. **In parallel** (if compute is available), kick off RF-DETR-Base training.
5. Rewrite `parser.py` + `aggregator.py` + `types.py` for the new schema. These don't depend on a trained model.
6. Wire `PaddleOCR-VL` engine; smoke-test on a handful of crops.
7. End-to-end pipeline run on a held-out video.
8. Iterate: fine-tune PaddleOCR-VL LoRA if needed; ensemble detectors with WBF if both look good.
9. Final report generator + dedup pass.
10. Lock final config, freeze checkpoints, write a short submission README.

---

## 12. Killer feature (STRETCH GOAL — only if the core metric is solid)

> **Scope guard.** This is *optional* upside, attempted **only after** the
> graded pipeline (detection → barcode → dedup → substantive fields → CSV) is
> solid. It does **not** feed the technical metric. It is a deliberate bet on
> the *non-metric* finals criteria — the organizers repeatedly say they value
> "maturity of approach", "scalability", and "applicability in the business
> process", and the task's own business framing is *shelf-compliance
> automation* ([`hackathon/task.md`](./hackathon/task.md) §1,
> [`hackathon/briefing.md`](./hackathon/briefing.md) §7–8). Do not let this
> displace core work.

**Idea.** On top of per-tag recognition, emit a lightweight **shelf-analytics
layer**: for each detected/recognized product, report **how many facings
(visible front units) are on the shelf**, plus optional empty-slot / gap
flags. This turns "a CSV of tags" into "a shelf-state report" — the actual
business outcome Lenta described (faster shelf audits, digital monitoring).

**Why it fits this repo.** We already detect, track, and dedup objects per
frame. Counting visible facings is mostly an aggregation head on top of
existing detections + the tracker, not a new model. Gap/OOS is a small extra
detector. The data backing exists — see
[`data/datasets-research.md`](./data/datasets-research.md): **Locount** §4.4
(localization + counting), **SKU-110K** §4.1 (dense facing detection),
**Gap Detection / ROSCH** §4.7/§4.15 (OOS), and the modular MVP architecture
§6/§8. Honest scoping (from that research): a single RGB pass gives
**visible facings**, *not* true stock-depth — label the metric accordingly
("visible facings / estimated stock"), which is itself the kind of
limitation-awareness the organizers reward.

**Minimal plan (in priority order, each independently shippable):**
1. **Facings count per product** — group deduped detections by shelf
   row + x-range; report `visible_facings` as an extra, clearly-separate
   output column / panel (never inside the graded 29-column CSV).
2. **Gap / empty-slot flag** — small detector or heuristic on row gaps;
   surfaced in the UI overlay, not the scored CSV.
3. **Shelf-state summary** in the UI: per-section product list with price +
   facings + gap, the "applicability" story for the presentation.

**Hard constraints (same as core):** local-only inference, lightweight enough
for the edge scenario, every dataset used disclosed in the README. Keep this
output **physically separate** from the graded CSV so it cannot corrupt the
metric. Tracked here as future work; not on the §11 critical path.

---

## Sources (verified May 2026)

- [RF-DETR (ICLR 2026) — Roboflow GitHub](https://github.com/roboflow/rf-detr)
- [YOLO26 — Roboflow blog](https://blog.roboflow.com/yolo26/)
- [Best Object Detection Models 2026 — Roboflow](https://blog.roboflow.com/best-object-detection-models/)
- [PaddleOCR-VL Hugging Face](https://huggingface.co/PaddlePaddle/PaddleOCR-VL)
- [PaddleOCR-VL paper (arXiv 2510.14528)](https://arxiv.org/abs/2510.14528)
- [PaddleOCR-VL 1.5 deep dive — Towards AI, Apr 2026](https://medium.com/@mustafa.gencc94/paddleocr-vl-1-5-a-deep-dive-into-the-0-9b-model-that-outperforms-gpt-4o-on-document-parsing-c93bac97ac1f)
- [Qwen3-VL repo](https://github.com/QwenLM/Qwen3-VL)
- [BoT-SORT repo](https://github.com/NirAharon/BoT-SORT)
- [Ultralytics tracking docs](https://docs.ultralytics.com/modes/track)
