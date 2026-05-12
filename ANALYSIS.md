# ANALYSIS.md — Price-Tag Recognition Hackathon (Lenta Tech Life 2026)

> Written cold, no sugarcoating. The repo is a runtime skeleton, not a solution.

## TL;DR

What we have is a **runtime scaffold for the inference path only**. There is no data pipeline, no training code, no metrics, no logging, no reproducibility tooling, and the default detector (`yolo11n.pt`) is COCO-pretrained and filtered to `classes=[0]` (`person`) — so out of the box the pipeline produces zero price-tag detections. The OCR step defaults to `noop`; the only Cyrillic-capable backend (PaddleOCR) is wired with an API surface that is **stale relative to PaddleOCR 3.x**. The parser is single-currency / single-price and has no concept of a "regular price + loyalty-card price" split, which is the dominant tag layout in Lenta stores. The sample output is in EUR with a decimal point — strong signal that the author wrote this as a generic demo, not for the Russian retail domain.

We can keep the module boundaries (`detector → rectifier → ocr → parser → aggregator`) — they are sensible. **Everything inside the modules needs rework, plus we need to add roughly half the codebase from scratch:** data prep, video-level split, detector training config, OCR/VLM training or fine-tuning recipe, metrics, logging, end-to-end eval, and a real video inference report.

---

## 1. What's already implemented and working

The pieces that exist and are sound enough to keep:

| Component | File | State |
|---|---|---|
| Config loading via YAML + frozen dataclasses | `src/price_tag_pipeline/config.py` | Solid, keep. Type-safe access pattern is nice. |
| Pipeline orchestration loop | `src/price_tag_pipeline/pipeline.py` | Structure is OK. Some bugs (see §3). |
| Aggregator: per-track weighted voting | `src/price_tag_pipeline/aggregator.py` | Algorithm is reasonable for one-field voting. Will need extension for structured records. |
| Crop sharpness gate | `src/price_tag_pipeline/quality.py` | Trivial but useful. |
| Aggregator + parser unit tests | `tests/` | Exist. Cover the easiest two modules. |
| Runtime profile concept (`fast` / `balanced` / `hq`) | `configs/*.yaml` | Good idea, but the differences between profiles are timid — see §3. |
| Modular separation (detector / rectifier / ocr / parser / aggregator) | layout | Good architecture. Keep the boundaries. |

That's it. Eight files of useful structural code, two trivial tests, three timid configs.

---

## 2. Architectural decisions: good vs questionable

### Good
- **Module boundaries.** Detector, rectifier, OCR, parser, aggregator are cleanly separated. Keep this — we will swap implementations behind those interfaces.
- **Frozen dataclasses for config** instead of dict-fishing everywhere.
- **Track-level aggregation by `track_id`.** Right call. Cross-frame voting is the only way to get high accuracy on a single tag seen N times.
- **JSONL append output** rather than building a list and dumping at the end. Crash-safe.

### Questionable / wrong-on-purpose
- **OCR default backend `noop`.** Ships green — the pipeline runs, produces nothing, and `len(preds)=0` is easy to miss. Should be a hard error or default to a real backend.
- **PaddleOCR API call (`use_angle_cls=True`, `self._ocr.ocr(image, cls=True)`).** That's PaddleOCR 2.x. The 3.x API renamed `use_angle_cls` → `use_textline_orientation` and changed the return tuple shape. Will break against current pip install of `paddleocr`.
- **Single flat `ParsedPrice`** with one `value` field. Russian price tags routinely show **two prices** (regular vs loyalty-card), product name, weight/volume, and price-per-unit. The current schema can't represent the task output.
- **Rectifier kills color** (BGR → GRAY → CLAHE → BGR). Color is *signal* in Russian price tags — yellow/red regions mark the promo price. Throwing it away hurts both downstream OCR and the regular/promo split task.
- **`rotate_vertical_tags`** is a 90° rotation toggle — too coarse. Price tags are rarely *exactly* vertical; the real problem is perspective from a robot driving past, which needs a 4-point perspective unwarp, not a 90° flip.
- **Tracker config points to `ultralytics/cfg/trackers/bytetrack.yaml`** — only works if Ultralytics is installed, and we lose the ability to tune `track_buffer`/`match_thresh` per profile. Should ship local tracker YAMLs.
- **Currency detection regex** parses RUB/EUR/USD. We are in a Russian chain. There is no scenario where EUR/USD appears on a Lenta shelf. Pre-committing to RUB removes a class of false positives.

---

## 3. Weak spots, gaps, bugs

Real bugs and serious holes. Roughly in order of damage they cause:

### B1. Hardcoded `fps = 30.0` in `detector.py:29`
Every `timestamp_s` in the output is wrong for any video that isn't 30 FPS. Robot footage is often 25 or 60. Must read `cv2.VideoCapture(...).get(cv2.CAP_PROP_FPS)` or take FPS from the Ultralytics result metadata.

### B2. Default `yolo11n.pt` + `classes: [0]` = "detect people"
`yolo11n.pt` is COCO-pretrained. Class 0 is `person`. Until we train a real detector, this filter must be removed or the pipeline must refuse to start without a `price_tag` checkpoint. This is the single highest-confusion footgun in the repo.

### B3. No detector training pipeline
There is **zero** code for training the detector. No dataset adapter, no YOLO YAML generator, no augmentation pipeline, no validation split. Comment in the README admits this ("Train detector on real shelf footage"). Without this we cannot compete.

### B4. No video-level train/val split
Frame-level splits leak: adjacent frames are near-duplicates. The split must be at the **video** level, ideally **stratified by store/aisle** if metadata is available. None of this exists.

### B5. Parser can't represent a Lenta price tag
A real Lenta tag has at least: regular price, loyalty-card price, product name, weight/volume, price-per-unit. The schema (`ParsedPrice`) has one `value`. The output (`FinalPrediction`) has one `price`. Even a perfect OCR couldn't fill the data we're being graded on.

### B6. Parser regex misses real-world formats
- `129⁹⁹` (superscript kopecks — extremely common in Russian price tags)
- `1 299,99` (thousand-separator space)
- `129.` (trailing dot)
- `12999` glued by OCR with no separator
- Multi-line OCR output where the major-rouble and minor-kopeck digits are on different lines

### B7. Aggregator votes on the exact `normalized_price` string
If OCR reads `129.99` in some frames and `12999` in others (because the comma is missed), they become different vote keys and split the count, failing both `min_observations_per_track` and `min_final_confidence`. Need fuzzy keying / value-based bucketing with rounding tolerance.

### B8. Tracker re-emission and double-counting
Once `flush_expired` finalizes a track, that `track_id` is removed from the aggregator. ByteTrack/BOT-SORT can re-emit the same numeric ID later (especially after re-ID gaps). This produces **duplicate final predictions for the same physical tag**. We need geometric/embedding dedup at the report-build stage (cross-track matching by IoU of last-seen bbox and content similarity).

### B9. Rectifier drops color
See §2. CLAHE-on-grayscale is the right idea for OCR contrast but throwing the color channel away closes the door on color-based promo-vs-regular detection and on VLM-based extraction.

### B10. `min_observations_per_track` is too aggressive for fast camera pans
With `balanced` setting `min_observations_per_track: 3`, any tag visible for only 1–2 frames (likely at shelf-edge or fast traversals) is dropped entirely. Need an adaptive policy: fewer-observation tags allowed if their detection confidence × sharpness is high.

### B11. No motion-blur handling
The robot moves; many frames have motion blur. We gate them out by sharpness, but never search for "the sharpest frame in this track's window". Easy win: pick top-K sharpest crops per track and OCR those, not the first ones that pass the gate.

### B12. PaddleOCR 3.x API mismatch
See §2. Will crash on import or `ocr()` call.

### B13. Tesseract confidence is a constant `0.55`
`image_to_string` gives no confidence; `image_to_data` does (per-word). The constant 0.55 is meaningless and corrupts the voting score downstream.

### B14. No GPU device selection
`device: null` → Ultralytics auto-pick, OK in practice. But there's no way to pin a device for multi-GPU rigs or to verify CUDA usage from config.

### B15. No metrics anywhere
- No mAP@[.5:.95] for detection
- No CER / WER for OCR
- No end-to-end accuracy (% of tags with all fields correct)
- No way to compare two checkpoints

Even if we train a great model, we have nothing to score it with.

### B16. No logging integration
No W&B, no TensorBoard, no MLflow. We will be flying blind during training.

### B17. No reproducibility tooling
- No `requirements.txt` pinning (only `numpy>=1.23`)
- No `pyproject.toml` / `uv.lock`
- No fixed seeds anywhere
- No Dockerfile
- No environment instructions

### B18. Tests cover the wrong things
Two tests, both for the easy pure-Python pieces. Nothing tests the detector adapter, rectifier geometry, OCR engine output shapes, or end-to-end smoke. Nothing protects us from regressing the pipeline.

### B19. Sample output is in the wrong domain
`sample_output.jsonl` emits EUR prices with a decimal point. This is a generic OCR demo signature, not a Lenta artifact. If anyone copies this format to seed downstream work, we will be evaluating against the wrong schema.

### B20. No structured product-name field
Product name is a graded field. Cyrillic OCR is finicky; we need a strategy (VLM end-to-end, or PaddleOCR with a Russian-language-model rescorer). None of that exists.

### B21. No CLI for evaluation
Only `run_price_tag_pipeline.py` for inference. We need at least: `prepare_data.py`, `train_detector.py`, `eval_detector.py`, `train_ocr.py` (or a config-only OCR setup), `eval_end_to_end.py`, `make_report.py`.

### B22. Aggregator's `source_frames` is filtered to the winning-price bucket
Cosmetic, but a reviewer might expect "all frames this track was observed in", not "frames whose OCR result matched the final answer". Document explicitly or rename.

### B23. `result.orig_img.copy()` on every frame
Wasted memory for long videos. Ultralytics already manages the buffer; we copy defensively for downstream rectification, but a cropped view + lazy-copy in rectifier would be cheaper.

### B24. `_PRICE_RE` accepts `\d{1,5}` major digits → max 99 999
Matches the parser's `max_price=99999.99`. Premium SKUs (electronics, alcohol) can exceed that. Easy to bump but currently silently truncates 6-digit prices.

---

## 4. Risks that cost ranking

Ranked by expected impact on leaderboard position:

| # | Risk | Why it costs us | Mitigation lane |
|---|---|---|---|
| R1 | **Underpowered detector** — using stock YOLO11n on small objects | mAP collapse on small price tags. Public bench: YOLO11n ≈ 39.5 mAP COCO; small-object recall is much worse. | Train YOLO11x / RT-DETR-L / Co-DETR-Swin on a tiled input at high resolution. |
| R2 | **Cyrillic OCR quality** — PaddleOCR Russian model is mid-tier | Wrong digits = wrong price = zero score for that tag. | Compare PaddleOCR-RU vs Qwen2.5-VL-7B/72B vs InternVL3 vs Donut-fine-tuned. End-to-end VLM likely wins on structured extraction. |
| R3 | **No regular/promo split** | Likely a graded field. Reporting the wrong price (regular shown but card-price expected, or vice versa) drops every tag. | VLM with structured JSON prompt, or two-region OCR with color/geometry heuristics. |
| R4 | **Video-level leakage** | Inflated val metrics → wrong checkpoint shipped → poor leaderboard. | Strict video-level GroupKFold split, ideally with stratification by store. |
| R5 | **Duplicate predictions from track-ID re-emission** | If grading penalizes false positives, every duplicate is a miss. | Cross-track dedup pass on the final report (IoU + price equality + temporal window). |
| R6 | **Hardcoded FPS** | Wrong timestamps → ambiguous tag identity → de-dup failures. | Read FPS from container. |
| R7 | **Reproducibility / single-machine drift** | Friend's environment ≠ rented A100 environment → silent failure mid-training. | Dockerfile + pinned `uv.lock` + smoke tests. |
| R8 | **Lack of metrics during training** | We can't pick the best checkpoint. | W&B + per-epoch mAP / CER / E2E. |
| R9 | **Augmentation mismatch** | Default Ultralytics aug doesn't model motion blur / glare / Cyrillic. | Domain-specific Albumentations pipeline. |
| R10 | **6-digit prices clipped** | Edge case but every clipped tag is a guaranteed miss. | Widen regex + parser bounds. |

---

## 5. Prioritized improvement list

### P0 — critical, blocks competing
- **P0.1** Build `prepare_data.py`: parse the annotation format (COCO/YOLO/CVAT — TBD on data inspection), validate integrity (boxes inside frame, no empty masks, no NaNs), emit YOLO-format dataset YAML.
- **P0.2** **Video-level** train/val split (GroupKFold by `video_id`, stratified by store/aisle where metadata exists). Store split manifest in repo.
- **P0.3** Detector training entrypoint with Hydra (or plain argparse + YAML) — `train_detector.py`. Backbone-switchable (YOLO11x, RT-DETR-L, RT-DETR-X, Co-DETR-Swin). Resume / multi-GPU / mixed-precision out of the box.
- **P0.4** Replace the EUR/decimal-point sample output with a proper Lenta schema: regular_price, loyalty_price, product_name, weight, weight_unit, price_per_unit, currency=RUB. Update `FinalPrediction` and `ParsedPrice` accordingly.
- **P0.5** Fix `fps` (read from container) and remove the `classes=[0]` hardcode (or force `classes: [price_tag_class_id]` after we train, default to None → detect all classes the trained model knows).
- **P0.6** OCR backend selection that actually works against PaddleOCR 3.x; add a Qwen2.5-VL / dotOCR backend (whichever wins on a held-out sample).
- **P0.7** Detection metric (`eval_detector.py` → mAP via pycocotools or torchmetrics).
- **P0.8** End-to-end metric (`eval_end_to_end.py` → per-field exact-match + Levenshtein for text fields).

### P1 — important, big gains
- **P1.1** Augmentation pipeline tuned for our domain: motion blur (high σ), Gaussian noise, brightness/contrast jitter, glare overlay, partial occlusion, small random rotation, mosaic+mixup at low prob (mosaic hurts small-object recall above ~0.3). Albumentations integration for Ultralytics.
- **P1.2** Tiled inference (SAHI) at test time — price tags are small; tiling typically gains 3–8 mAP points on small-object datasets.
- **P1.3** Real perspective rectification: 4-point homography from detected tag corners (or a small corner-keypoint head) instead of axis-aligned crop+rotate. Boosts OCR by a lot.
- **P1.4** Cross-track deduplication pass: cluster final predictions by (IoU, price, time window) before writing the report.
- **P1.5** Motion-blur-aware crop selection: pick top-K sharpest frames per track instead of "first ones to pass the gate".
- **P1.6** Logging (W&B). Log: dataset version, config, training curves, validation predictions (image grid every N epochs), failure-mode crops.
- **P1.7** Adaptive aggregation: lower `min_observations_per_track` when det_conf × sharpness is high, to recover fast-pan tags.
- **P1.8** Tracker tuning: ship local `bytetrack.yaml` and `botsort.yaml`. Sweep `track_buffer` and `match_thresh`.
- **P1.9** Fuzzy aggregator keying: bucket prices that round to the same value (within `±0.5` of the lower-quality digit).
- **P1.10** Smoke tests: 5-batch training step, 1-video inference, schema validation of the JSONL report.

### P2 — nice to have, time permitting
- **P2.1** Color-region heuristic for promo-vs-regular price (yellow/red mask → "this region is the promo price").
- **P2.2** TensorRT export for the chosen detector; benchmark FPS in the final report.
- **P2.3** Self-supervised pretraining of the OCR/VLM head on unlabeled shelf frames (if we have them).
- **P2.4** Ensembling: average two detectors (e.g. RT-DETR-L + YOLO11x) at inference via WBF.
- **P2.5** Confidence calibration (temperature scaling on a held-out fold) so `final_confidence` is interpretable for downstream filtering.
- **P2.6** Dockerfile + GitHub Actions CI running smoke tests + unit tests + lint.
- **P2.7** Streamlit / Gradio demo for the jury (if there's a "best demo" prize).

---

## 6. What I'd touch carefully

These are the few things the existing author got right; rewriting them buys nothing and risks introducing bugs:

- `config.py` — keep the dataclass-loading pattern. Extend it for new sections (training, augmentation, eval, report) but don't replace it with Hydra-only.
- `aggregator.py` core voting algorithm — extend it to multi-field voting (one vote per field), don't rewrite from scratch.
- `quality.py:laplacian_sharpness` — fine.
- Module boundaries in `pipeline.py` — keep them; rewrite the *bodies*.

---

## 7. Open questions for the user (before Stage 2)

1. **Annotation format and data path.** The repo has no data and no path placeholder. Where will the dataset land? CVAT / COCO / YOLO / custom?
2. **Output schema specified by the organizers.** Do we have a sample of the expected submission JSON/CSV? Without it I'm guessing the fields. Strong inference: regular_price, loyalty_price, product_name, weight, weight_unit, price_per_unit — please confirm or paste the spec.
3. **Are there per-video metadata fields** (store ID, aisle, time-of-day, camera ID) we can use to stratify the val split?
4. **Compute access.** "We can rent an A100/H100" — what's the workflow? SSH to a cloud box, vast.ai, or a managed notebook? Affects how I script training.
5. **Time budget.** How many days to the deadline? Determines whether we go all-in on a VLM (longer but stronger) or play it safer with detector + PaddleOCR + parser.

I'll keep moving on Stage 2 (STRATEGY.md) with my best assumptions and flag every place where I'd want your input.
