# Detector fine-tuning report — what worked, and the camera profile

Date: 2026-05-17. Purpose: hand-off notes for fine-tuning a strong
**pretrained** price-tag detector on our Lenta robot-video domain. This is the
curated, reusable distillation of the local experiments; the raw working
notes/runbook live on the `worktree-experiments` branch
(`docs/runbooks/detector-experiments.md`), the per-run numbers in
`runs/ledger.csv` there.

> **Read this first.** Every number below is from a **single fold** (5 videos,
> ~61 labeled frames, 274 boxes; held-out video `25_2-10`, 9 frames, 56 GT).
> It is **directional, not a benchmark** — mAP is unreliable at this data
> size. Treat the *rankings and effects* as the signal, not the absolute
> values. The dominant quality lever is **more data**, not architecture or
> hyper-tuning; spend effort on the wider dataset + hard negatives, fine-tune
> incrementally on top of the strong pretrained model.

---

## 1. TL;DR — fine-tuning recipe that the evidence supports

1. **Resolution is the single biggest lever.** Train and infer at **imgsz ≥
   1536** (ideally 1536→1920, or tiled/SAHI inference on the 4K frame). Tags
   are tiny: ~3–5 % of frame width on a 3840-px frame. Going 1280→1536 on
   yolo11s lifted held-out recall **0.48 → 0.82**.
2. **One class only** (`price_tag`). Do not fold product/package classes into
   the detector that feeds OCR — that is exactly the false-positive source.
3. **Pick the confidence operating point from the model's own
   recall/precision-vs-conf curve**, recall-first but FP-aware (every FP is an
   extra OCR call + dedup risk). Expect to retune per model — see §3.
4. **Windows: `workers=0` is mandatory** (DataLoader deadlock with
   `workers>0`, intermittent — cost it us a full hung run). On the tiny set it
   has zero throughput cost; revisit only if you move to Linux/large data.
5. **Validate by leave-one-video-out + visual QA**, never by train
   loss/Ultralytics-mAP alone. mAP lied here: s@1536 had the *best* recall but
   a *worse* mAP than s@1280 (it floods low-conf boxes).
6. **Augment for THIS camera** (barrel distortion + small-sensor + moving
   robot). Concrete albumentations block in §5 — this is the part that bridges
   a pretrained model's domain to ours.
7. **The remaining error is data, not model.** Hard-negative mining from the
   unlabeled videos + the wider dataset will move the needle far more than any
   architecture change.

---

## 2. Data & domain reality

- 5 videos, ~61 labeled frames (`49_5`=25, `26_12-20`=15, `25_12-20`=10,
  `25_2-10`=9, `43_15`=2), 274 boxes, one class.
- Source: 3840×2160, ~20 FPS, **strong barrel-distorted** robot footage
  (curved shelves — see §4).
- Frames are extracted only at annotated timestamps → the detector only ever
  trains on labeled frames. False-positive discovery needs inference on the
  *unlabeled* frames/videos (hard-negative loop), not the labeled val set.
- Validation = leave-one-video-out. Single-fold val (1 video) is noisy;
  `43_15` (2 frames) is not a usable fold on its own. Report mean ± spread
  across the 5 folds for any real decision.

---

## 3. What worked / didn't (empirical, fold 0, held-out `25_2-10`)

Operating point = fixed conf + IoU-0.5 greedy matching (our `eval_qa.py`), not
Ultralytics' conf≈0.001 PR-mAP.

| Run | imgsz | conf | recall | precision | FP | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|---|
| yolo11n builtin | 1280 | 0.05 | 0.589 | 0.094 | 320 | 0.294 | 0.150 |
| yolo11n builtin | 1280 | 0.40 | 0.375 | 0.223 | 73 | — | — |
| yolo11s builtin | 1280 | 0.05 | 0.482 | 0.117 | 203 | 0.303 | 0.161 |
| **yolo11s builtin** | **1536** | 0.05 | **0.821** | 0.051 | 853 | 0.199 | 0.098 |
| yolo11s builtin | 1536 | 0.25 | 0.607 | 0.101 | 304 | — | — |
| yolo11s builtin | 1536 | 0.40 | 0.500 | 0.123 | 200 | — | — |
| yolo11m builtin | 1536 | 0.05 | 0.554 | 0.144 | 185 | 0.284 | 0.163 |
| yolo11m builtin | 1536 | 0.40 | 0.161 | 0.750 | 3 | — | — |
| yolo11m builtin | 1536 | 0.55 | 0.036 | 1.000 | 0 | — | — |

**Read-outs that transfer to the fine-tune:**

- **imgsz 1536 ≫ 1280** for recall (small tags). This is the highest-value
  knob. If the pretrained model supports it, fine-tune and infer at ≥1536;
  consider SAHI/tiling so far/peripheral tags survive.
- **Small models (n/s) flood FP; bigger/better models trade recall for
  precision sharply with conf.** yolo11m goes P=0.75 @c0.40 but recall craters
  to 0.16. A *strong pretrained* model should beat this whole frontier — the
  point is the *shape*: you will still need a conf sweep to place the operating
  point, recall-first, because FP feed OCR.
- **mAP is not your selection metric here.** s@1536 (best recall) has the
  worst mAP. Select by held-out recall/precision at the deployed conf + visual
  QA, aggregated over folds.
- **Early stopping**: patience 40 → runs stopped ~45–85 epochs; best epoch was
  often early (~45). With the wider dataset, raise patience and epochs.
- **Optimizer**: Ultralytics `optimizer=auto` picked **AdamW lr≈0.002,
  momentum≈0.9**, `cos_lr=True` — a sane default to keep for fine-tuning;
  consider a lower lr0 for fine-tuning a strong checkpoint (e.g. 5e-4) and a
  short warmup.

---

## 4. Camera profile (from `example_undistort.py`)

```
image size      : 3840 × 2160  (16:9)
sensor diagonal : 16.0/2.8 ≈ 5.714 mm   → 1/2.8" sensor
focal length    : 2.8 mm                → wide-angle, fixed
distortion (Brown–Conrady, [k1,k2,p1,p2,k3]):
                  k1=-0.276  k2=0.060  p1=0.0084  p2=-0.0016  k3=-0.0044
derived intrinsics: fx≈fy≈2159 px, cx=1920, cy=1080
field of view   : ≈ 83° H × 53° V (≈ 90°+ diagonal)
```

Interpretation:

- **k1 = −0.276 → strong barrel distortion.** Straight shelf edges bow
  outward; this is clearly visible in the QA renders. Periphery tags are
  geometrically warped and shrunk.
- Small **tangential** terms (p1,p2) → slight lens decentering (asymmetry
  left/right).
- 1/2.8" sensor + 2.8 mm lens → wide FOV, **fixed focus** (near *and* far
  tags can be soft), and the small sensor means **noticeable noise in dim
  aisles**.
- **Geometric-space consistency rule:** the provided `DistortionCorrector`
  (`cv2.initUndistortRectifyMap` + `getOptimalNewCameraMatrix` + ROI crop)
  rectifies frames. Train and infer in the **same** space. Two valid options:
  - **(A) Raw distorted (recommended, simplest):** train + infer on raw
    frames (as the current labeled set is). Keep the barrel — the model
    learns it. Add mild distortion *jitter* (below) for cross-camera
    robustness.
  - **(B) Undistort everywhere:** run `DistortionCorrector` on every frame
    before train *and* inference (note: ROI crop changes resolution/aspect,
    re-tune imgsz). Only worth it if OCR-stage crop rectification clearly
    benefits. Never mix A and B.

---

## 5. Camera-matched augmentation block (the bridge to our domain)

A strong pretrained detector was almost certainly trained on different optics.
Fine-tune with augmentations that **simulate this camera and these store
conditions** so the model adapts to our domain, plus distortion *jitter* so it
generalizes across stores/cameras in the wider dataset.

### 5.1 Built-in Ultralytics aug — kept as-is (these worked)

From `price_tag_pipeline/training/augmentation.py` (`UltralyticsAugConfig`),
already domain-tuned and used in the runs above:

```
hsv_h=0.010  hsv_s=0.55  hsv_v=0.35      # vary lighting, KEEP red/yellow tag signal
degrees=3.0  translate=0.07  scale=0.35  shear=1.0  perspective=0.0008
flipud=0.0   fliplr=0.5                   # never vertical-flip (inverts price text)
mosaic=0.5   mixup=0.0   copy_paste=0.0   erasing=0.4
```

- **Keep `flipud=0`, small `degrees`** — price text orientation matters.
- **Sweep `mosaic ∈ {0.0, 0.3, 0.5}` + `close_mosaic` last ~10 epochs.** The
  config itself warns mosaic >0.3 hurts small-object recall, and our tags are
  tiny — this is a real knob, not a default.
- **Keep HSV jitter gentle** — the detector keys on the **red price digits**
  (confirmed by visual QA: it reliably hits red-digit tags). Aggressive
  hue/sat shifts would destroy that signal.

### 5.2 Heavy domain albumentations — fixed, recommended

`price_tag_pipeline/training/augmentation_albu.py` (pass `--use-albu`) was
broken on the installed stack (Albumentations 2.x renamed args; Ultralytics
8.4.x passes a new `transforms=` kwarg). **Both are now fixed**
(version-aware constructor selection + `*args/**kwargs` subclass). Recommended
families (robot-motion + store-light + small-sensor realistic):

| Transform | Why (this camera/domain) | Strength |
|---|---|---|
| MotionBlur | robot moves along shelves | p≈0.35, kernel ≤25 |
| Defocus / GaussianBlur | 2.8 mm fixed-focus → soft near/far tags | p≈0.10 |
| ISONoise / GaussNoise | 1/2.8" small sensor, dim aisles | p≈0.25 |
| RandomBrightnessContrast | aisle-to-aisle lighting | p≈0.45 |
| CLAHE | low-contrast / shadowed tags | p≈0.10 |
| RandomShadow / RandomSunFlare | plastic shelf-cover glare, ceiling lights | p≈0.15 / 0.10 |
| HueSaturationValue (gentle) | white-balance drift, **preserve red/yellow** | p≈0.20, small limits |
| Perspective | viewing angle from moving robot | p≈0.30 |
| HorizontalFlip | shelf-direction symmetry | p≈0.5 |
| CoarseDropout | carts/hands/partial occlusion | p≈0.15 |

### 5.3 Add: camera-matched geometric/codec aug (new — do this)

Tie these to the measured camera so fine-tuning closes the optics gap:

- **Lens-distortion jitter** — `albumentations.OpticalDistortion`
  (`distort_limit≈(-0.10, 0.10)`) and/or `GridDistortion`
  (`num_steps=5, distort_limit≈(-0.1,0.1)`), p≈0.2–0.3. Rationale: our k1 is
  ≈−0.28; jittering distortion around the operating point makes the detector
  robust to (a) residual distortion if you undistort, and (b) per-store
  camera/zoom variation in the wider dataset. If using option **5.4**, this is
  the cheap approximation of it.
- **JPEG/video codec artifacts** — `ImageCompression(quality≈40–85)`, p≈0.3.
  Frames come from compressed video; the pretrained model likely saw clean
  stills.
- **Vignetting / corner falloff** — wide lens + small sensor darkens corners
  where peripheral tags already suffer from barrel warp. A mild radial
  darkening aug (or `albumentations` `RandomToneCurve`/custom vignette),
  p≈0.15.
- **Mild chromatic aberration** at edges (wide lens) — optional, low p.
- Keep geometric augs **bounded**: the acceptance rule is empirical — keep an
  augmentation only if held-out (leave-one-video-out) recall or downstream
  crop readability improves; drop it otherwise.

### 5.4 Optional, highest-fidelity: exact-camera resampling

Instead of generic `OpticalDistortion`, synthesize training variants by
**re-projecting** with the real `K` and `CAM_DISTORT_COEFFS` (perturb k1/k2 ±
10–20 %, small cx/cy/focal jitter) using the same `initUndistortRectifyMap`
math as `example_undistort.py`. This produces physically-faithful distortion
augmentation matched to this camera family. Worth it once the wider dataset is
in and you want the last few points; not required for the first fine-tune.

---

## 6. Failure modes from visual QA (drives hard-negative mining)

Looking at rendered GT-vs-pred frames (red=missed GT, green=TP, yellow=FP):

- **What the model gets right:** high-contrast **red-digit price tags** are
  reliably detected — the core signal is learnable even from 274 boxes.
- **Systematic false positives** (the precision killer): wine-bottle
  labels/necks, product packaging, **white country shelf-cards** (e.g.
  "Испания"), the **yellow Lenta floor banner**, and glare/reflections on
  plastic shelf strips.
- These are *systematic object classes*, so the highest-leverage data move is
  **hard-negative mining**: run the fine-tuned model at low conf on the
  unlabeled videos/frames, harvest the FP-heavy crops/frames, add as negatives
  / correct labels, retrain. This + the wider dataset is where the precision
  comes from — not threshold tuning (yolo11m still had 73 FP at the conf where
  recall was already gone).

---

## 7. Validation methodology to reuse

- **`eval_qa.py`** (on `worktree-experiments`): operating-point
  recall/precision at a chosen conf + IoU-0.5 matching, plus QA renders
  (worst-frame-first, thick boxes, downscaled to be legible from 4K) and an
  append-only `runs/ledger.csv`. Use it per checkpoint.
- **Leave-one-video-out**, report mean ± spread; do not trust a single fold.
- **Look at the frames.** mAP/loss were actively misleading at this scale; the
  visual QA is what revealed the FP taxonomy and that imgsz was the lever.

---

## 8. Environment & process gotchas (so the fine-tune doesn't lose a day)

- **Windows + Ultralytics/PyTorch: `workers=0`.** `workers>0` intermittently
  deadlocks the DataLoader at training start (and in `model.val`). Set 0 in
  both train and val. Tiny-data cost is nil.
- **CUDA torch**: `torch 2.6.0+cu124` on RTX 4070 Ti (12 GB) works; yolo11m @
  imgsz 1536 needs `batch≈4` to fit; yolo11s @ 1536 `batch 8` ≈ fits.
- **Unicode-safe image IO**: go through `price_tag_pipeline.cv_io`
  (`imread/imwrite`), never `cv2.imread/imwrite` directly — the build machine
  sits under a Cyrillic Windows user path; bare cv2 silently fails.
- **Path discipline on Git-Bash/Windows**: the venv `python.exe` cannot open
  MSYS `/e/...` paths — pass Windows `E:/...` paths to anything handed to
  Python; use MSYS paths only for bash file ops.
- **API drift handled**: `augmentation_albu.py` is now version-aware for
  Albumentations 1.x↔2.x arg renames and Ultralytics 8.4.x `Albumentations`
  signature. `--use-albu` works again.
- Killing a background job does **not** kill Git-Bash grandchildren on
  Windows — orphaned trainers will fight for the GPU. Kill the python/bash
  tree explicitly when restarting a queue.

---

## 9. What NOT to do

- Don't pick checkpoints by train loss or Ultralytics mAP on this data size.
- Don't over-tune hyperparameters on 61 frames — diminishing returns; the
  wider dataset + hard negatives dominate. Fine-tune incrementally on the
  strong pretrained model; don't rewrite the pipeline to chase a number.
- Don't add product classes to the graded detector.
- Don't undistort for training while inferring on raw frames (or vice-versa).
- Don't raise conf to "fix" precision at the cost of recall before exhausting
  hard negatives — recall feeds OCR; missed tags are unrecoverable downstream.
