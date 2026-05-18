# Guide — annotate & extend the dataset with CVAT

This is the **friend-facing runbook**. Follow it top to bottom. Every
command is copy-paste; every click is spelled out. If you only read one
thing, read **§Pipeline at a glance** and the **§Cheat sheet**.

Goal: turn the organizers' raw videos into more training data. We slice
them into frames, **pre-label them automatically with our detector** (so you
*correct* boxes instead of drawing from scratch), load them into CVAT, you
fix them by hand, and a converter folds the result back into the project
with **zero pipeline change**.

> **Is hand-labelling allowed?** Yes — manual labelling **for training** is
> explicitly permitted (task.md §7) and must be *disclosed* in the README
> (volume, method, how used). It is **forbidden at inference**. Keep that line.

---

## Pipeline at a glance

```
            LAPTOP (CPU, no GPU needed)            │   GOOGLE COLAB (free GPU)
  ─────────────────────────────────────────────── │ ───────────────────────────
  STEP 1  slice_video_frames.py                    │
          videos → frames, drop robot-parked dupes │
                         │  upload cvat_video_frames/ to Drive
                         ▼                          │
                                          STEP 2  lenta_prelabel_colab.ipynb
                                          detector → a YOLO .txt per frame
                         ┌──────────────────────────┘
                         ▼  download the folder back (now has .txt files)
  STEP 3  frames_to_cvat.py → cvat_bootstrap_photos.py
          push frames + pre-boxes into CVAT
                         │
                         ▼
  ANNOTATE in the browser (http://localhost:8080) — fix/add boxes
                         │
                         ▼
  EXPORT  cvat_pull_pack.py   →   CONVERT  prepare_data.py
          → dataset folded into data/raw_photos, ready to train
```

STEP 2 is the only GPU step → it runs on Colab. Everything else is the
laptop. If your laptop *does* have a usable GPU you can run STEP 2 locally
(§3B) and skip Colab entirely.

---

## Part 0 — One-time setup

You need: **Docker Desktop** (running), **git**, and this repo checked out.
All commands assume:

* **cwd = this worktree:** `cd .../.claude/worktrees/annotation`
* **the worktree venv** for python: `.venv/Scripts/python.exe`
  (uv-managed; project rule: never global pip). On Linux/Mac it's
  `.venv/bin/python`.
* Replace **`***`** with the CVAT admin password (ask the repo owner; it is
  *not* written in any committed file).

If the venv is missing detector deps and you run STEP 2 **locally**:
`uv pip install --python .venv ultralytics huggingface-hub`. For Colab you
don't need them locally at all.

---

## Part 1 — Bring up CVAT

CVAT is vendored as a git submodule at **`third_party/cvat`** (pinned to
commit `c0f002237`, CVAT 2.64.1) and runs locally via its own Docker
Compose — no cloud (consistent with the inference constraint, task.md §10).

```bash
cd third_party/cvat
docker compose up -d            # first run pulls images (~minutes)
cd ../..                        # back to the worktree root for everything else
```

* Open **http://localhost:8080** and log in (`admin` / `***`).
* First account only, if it doesn't exist yet:
  `docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'`
* Pause / resume any time — state lives in Docker volumes (§Storage):
  * `cd third_party/cvat && docker compose stop` — pause
  * `cd third_party/cvat && docker compose up -d` — resume
* **NEVER** `docker compose down -v` — `-v` wipes every annotation volume.

Submodule clone failed on a flaky network? Shallow + pinned:
`git -c http.version=HTTP/1.1 clone --depth 1 --branch v2.64.0 https://github.com/cvat-ai/cvat third_party/cvat`
then `git submodule add --force https://github.com/cvat-ai/cvat third_party/cvat`.

---

## Part 2 — STEP 1: slice videos into frames (laptop)

Put the organizers' videos anywhere (e.g. the released set at
`real_data/dataset/<id>/<id>.mp4`, or a folder of new clips). Then:

```bash
.venv/Scripts/python.exe projects/price_tag_pipeline/scripts/slice_video_frames.py \
  --src "E:/Hackatons/lenta_tech_life_2026/real_data/dataset" \
  --every 2.0 --out cvat_video_frames
```

* `--every 2.0` — one frame every 2 seconds (the default; tune freely).
* **Auto orientation.** Different clips can come from different camera mounts.
  The slicer now defaults to `--rotate auto`, which keeps raw frames and lets
  STEP 2 score `none` / `cw` / `ccw` / `180` with the detector per scene. If a
  different orientation wins, STEP 2 rotates the scene images and clears stale
  `.txt` labels before writing new boxes. Use explicit `--rotate ccw|cw|180|none`
  only when you know the whole source folder has one fixed orientation.
* **Robot-parked de-dup is automatic.** The scan robot often stops; those
  runs of identical frames are dropped (compared against the last *kept*
  frame, so a long pause collapses to one frame, not every Nth).
  `--min-diff 0.02` is the default sensitivity; `--min-diff 0` keeps all
  frames; raise it to drop more aggressively.
* Single file instead of a folder: `--video path/to/clip.mp4`.

Output: `cvat_video_frames/<scene>/NNNNNN.jpg` + `slice_manifest.json`.
**One scene = one video** (becomes one CVAT task).

---

## Part 3 — STEP 2: pre-annotate with our detector

"Our detector" = the **OpenFoodFacts price-tag YOLO**
(`hf://openfoodfacts/price-tag-detection/weights/best.pt`) — the exact
weights the original auto-label notebook used and the fixed base of our
solution. Override with a fine-tuned checkpoint via `--model` / `MODEL_PATH`
once we have one.

### 3A — Google Colab (recommended — no GPU on the laptop)

The notebook **`notebooks/lenta_prelabel_colab.ipynb`** *is* this step.

1. Open https://colab.research.google.com → **File → Upload notebook** →
   pick `notebooks/lenta_prelabel_colab.ipynb`.
2. **Runtime → Change runtime type → T4 GPU → Save.**
3. In Google Drive (drive.google.com) create `MyDrive/lenta_prelabel/` and
   **drag the whole `cvat_video_frames/` folder** (from STEP 1) into it, so
   it lands at `MyDrive/lenta_prelabel/cvat_video_frames`.
4. In the notebook: **Runtime → Run all**. Approve the Drive-mount popup.
5. Wait for `DONE: N frames, M pre-labelled boxes`. The notebook wrote a
   `.txt` next to every `.jpg` **in that Drive folder**.
6. Download the folder back to the laptop (right-click → Download in Drive,
   or use CELL 4's zip) and replace the local `cvat_video_frames/` with it,
   so locally every `.jpg` now has a sibling `.txt`.

Nothing to edit in the notebook unless you want a different model (CELL 2).

### 3B — Local (only if the laptop has a decent GPU)

```bash
uv pip install --python .venv ultralytics huggingface-hub   # once
.venv/Scripts/python.exe projects/price_tag_pipeline/scripts/prelabel_frames.py \
  --frames-dir cvat_video_frames
```

Same result: a YOLO `.txt` next to each frame (empty `.txt` = "checked, no
tag" — a useful **negative**, keep it). Add `--device 0` for GPU, `cpu` to
force CPU (slow). `--model data/checkpoints/detector/best.pt` to use a
fine-tuned checkpoint.

---

## Part 4 — STEP 3: push frames + pre-boxes into CVAT (laptop)

```bash
.venv/Scripts/python.exe projects/price_tag_pipeline/scripts/frames_to_cvat.py \
  --frames-dir cvat_video_frames --out cvat_seeds_video
.venv/Scripts/python.exe projects/price_tag_pipeline/scripts/cvat_bootstrap_photos.py \
  --manifest cvat_seeds_video/manifest.json --user admin --password ***
```

* `frames_to_cvat.py` turns each scene's frames + `.txt` into a *CVAT for
  images 1.1* seed and a bootstrap manifest (every frame is uploaded —
  blank ones too, so confirming "no tag here" is recorded).
* `cvat_bootstrap_photos.py` creates one **image task per scene** in the
  **"Lenta price tags"** project, uploads the photos, and imports the
  pre-boxes. Re-running skips existing scenes; add `--replace` to re-import
  annotations into an existing scene **without** re-uploading photos.

---

## Part 5 — Annotate (browser)

1. http://localhost:8080 → project **"Lenta price tags"** → open a task →
   open its **Job** (boxes only show *inside a job*, not in list views).
2. Each price tag should have one tight `price_tag` box. The detector
   pre-drew them: **delete false boxes, fix loose ones, add missed tags.**
   Boxes only — there are no fields/attributes to fill (detector-first; the
   per-object "details" panel is intentionally off).
3. Shortcuts: `N` new box · drag handles to resize · `Del` delete ·
   `F` next frame · `D` previous · `Ctrl+S` save (save often).
4. A frame with genuinely no tag → just leave it empty and move on; that's
   a valuable negative.

These are **image tasks** (independent frames, no tracking) — the right
model here because we sample every 2 s. The tracking/interpolation mode is
explained in the appendix for the legacy video-task flow.

---

## Part 6 — Export from CVAT

In the task: **Actions → Export task dataset →** format
**"CVAT for images 1.1"**, **"Save images" → OFF** (we already have the
media). You get a zip with an `annotations.xml`. You can export every scene,
or let the puller pull straight from the running server (next step) — no
manual export needed.

---

## Part 7 — Convert to our format

One command pulls the validated scenes straight from CVAT, writes
single-class YOLO, zips it (to share), **and** folds it into the pipeline
raw layout:

```bash
.venv/Scripts/python.exe projects/price_tag_pipeline/scripts/cvat_pull_pack.py \
  --tasks 25_12-20,25_2-10,26_12-20,43_15,49_5 \
  --images-dir cvat_video_frames --set-name video_frames \
  --user admin --password *** --into data/raw_photos
.venv/Scripts/python.exe projects/price_tag_pipeline/scripts/prepare_data.py \
  --raw data/raw_photos --processed data/processed_photos
```

* `--tasks` = the scene names (one per video). `--images-dir` must be the
  STEP 1 frames root so the puller finds the source JPEGs.
* Output filenames are `<scene>__<frame>` so multi-video merges never
  collide. `--labels-only` makes a tiny zip (no images) to send the friend.
* `prepare_data.py` is **unchanged** — it ingests `data/raw_photos` exactly
  like the original 5 videos. The dataset is now extended; train as usual.

---

## Cheat sheet (the whole pipeline)

```bash
# 0. CVAT up
cd third_party/cvat && docker compose up -d && cd ../..

# 1. slice + dedup (laptop)
.venv/Scripts/python.exe projects/price_tag_pipeline/scripts/slice_video_frames.py \
  --src "E:/Hackatons/lenta_tech_life_2026/real_data/dataset" --every 2.0 --out cvat_video_frames

# 2. pre-label  ->  Colab: run notebooks/lenta_prelabel_colab.ipynb
#                   local: prelabel_frames.py --frames-dir cvat_video_frames

# 3. push to CVAT (laptop)
.venv/Scripts/python.exe projects/price_tag_pipeline/scripts/frames_to_cvat.py \
  --frames-dir cvat_video_frames --out cvat_seeds_video
.venv/Scripts/python.exe projects/price_tag_pipeline/scripts/cvat_bootstrap_photos.py \
  --manifest cvat_seeds_video/manifest.json --user admin --password ***

# 4. ...annotate in the browser...

# 5. export + convert
.venv/Scripts/python.exe projects/price_tag_pipeline/scripts/cvat_pull_pack.py \
  --tasks <scene1,scene2,...> --images-dir cvat_video_frames --set-name video_frames \
  --user admin --password *** --into data/raw_photos
.venv/Scripts/python.exe projects/price_tag_pipeline/scripts/prepare_data.py \
  --raw data/raw_photos --processed data/processed_photos
```

---

## Other inputs (same tools)

* **Your own store photos.** Skip STEP 1–2. Make an image task, upload the
  photos, draw boxes, export, then `cvat_import.py --xml <ann.xml>
  --images-dir <photos> --set <id>` → `prepare_data.py`. Shoot 4K,
  landscape, **don't pre-rotate**, focus on barcodes + hard cases (glare,
  glass, blur).
* **Friend / external dataset** (`friends_scene_1..4`, already loaded from
  `friends_labels/`): `cvat_from_external.py --csv
  friends_labels/all_candidates2.csv --model-backed --images-dir
  friends_labels/dataset_lenta --scenes 4 --out cvat_seeds_friends` →
  `cvat_bootstrap_photos.py --manifest cvat_seeds_friends/manifest.json`.
  Export each scene → `cvat_pull_pack.py` / `cvat_import.py`.

---

## Tool reference

All under `projects/price_tag_pipeline/scripts/`, run with the venv python.

| Tool | Step | Purpose |
|---|---|---|
| `slice_video_frames.py` | **1** | videos → frames every N s, drop robot-parked dupes |
| `prelabel_frames.py` | **2 (local)** | detector → a YOLO `.txt` per frame |
| `notebooks/lenta_prelabel_colab.ipynb` | **2 (Colab)** | same, on free GPU |
| `frames_to_cvat.py` | **3** | frames + `.txt` → CVAT image-task seeds + manifest |
| `cvat_bootstrap_photos.py` | **3** | create image task per scene, upload, import seeds (`--replace`) |
| `cvat_pull_pack.py` | **5** | pull validated scenes → single-class YOLO + zip, `--into data/raw_photos` |
| `prepare_data.py` | **5** | existing pipeline ingest — **unchanged** |
| `cvat_import.py` | alt | a single CVAT export → ours (auto: boxes→detector, fields→tags) |
| `cvat_from_external.py` | alt | friend candidates CSV / YOLO-txt → scene seeds |
| `cvat_prepare_task.py` | alt | print the bare `price_tag` label spec / task steps |
| `cvat_strip_attributes.py` | fix | strip attributes off the live label (kills details panel) |
| `train_detector_yolo.py` | — | fine-tune the OFF detector on our data (`experiments/finetune_openfoodfacts.yaml`) |

Pure XML build/parse is cv2-free and unit-tested (`tests/test_cvat.py`,
`tests/test_frame_sampling.py`, `tests/test_detector_model_path.py`).

---

## Where everything is stored

| What | Location |
|---|---|
| CVAT source (pinned) | `third_party/cvat` (submodule @ `c0f002237`, 2.64.1) |
| **Annotations + uploaded media** | Docker volume **`cvat_data`** → `/home/django/data` |
| Tasks / users / labels | Docker volume **`cvat_db`** (PostgreSQL) |
| Keys / logs / caches | `cvat_keys`, `cvat_logs`, `cvat_events_db`, `cvat_inmem_db`, `cvat_cache_db` |
| Sliced frames + STEP-2 labels | `cvat_video_frames/<scene>/*.jpg|*.txt` (gitignored) |
| STEP-3 seeds + manifest | `cvat_seeds_video/` (gitignored) |
| Imported photos / detector data | `data/raw_photos/{frames,annotations/labels}/<set>/` (gitignored) |
| Adapter code | `src/price_tag_pipeline/data/{cvat,frame_sampling}.py` + `scripts/` |

Backup (stop CVAT first):
`docker run --rm -v cvat_db:/v -v "$PWD:/b" busybox tar czf /b/cvat_db.tgz -C /v .`
(repeat for `cvat_data`).

---

## Troubleshooting

* **"I don't see any boxes."** You opened the task page, not its **Job**.
  Open the job. Also check label **Opacity** in the right panel; 4K frames
  make ~400 px boxes look tiny at fit-to-screen.
* **Frames are sideways / upside down / boxes look wrong.** Re-run STEP 1 with
  the default `--rotate auto`, then STEP 2 with `--orientation auto`. If a CVAT
  task already has wrongly oriented photos, delete that task and re-push it:
  `--replace` swaps only annotations, not uploaded images.
* **STEP 2 says `ultralytics not installed`.** Run it on Colab (§3A) or
  `uv pip install --python .venv ultralytics huggingface-hub`.
* **Colab can't find the frames folder.** Path in CELL 0 must match where
  you uploaded it: `MyDrive/lenta_prelabel/cvat_video_frames`.
* **Re-pushing without re-uploading photos.** `cvat_bootstrap_photos.py …
  --replace` clears + re-imports annotations only.
* **CVAT won't start / port busy.** `cd third_party/cvat && docker compose
  ps`; logs: `docker compose logs cvat_server`. First boot runs DB
  migrations (~1 min).
* **Frames look too similar / too sparse.** Tune STEP 1 `--every` and
  `--min-diff`, re-run STEP 1→3 (use `--replace` in bootstrap).
* **`cvat-sdk` missing.** `uv pip install --python .venv cvat-sdk`.

---

## Appendix — why image tasks, not tracking

CVAT also has a *video/interpolation* mode (a **Track** = one box
interpolated across keyframes, with a stable id per physical object). We
**deliberately don't use it here**: we sample one frame every 2 s, so
there's nothing to interpolate, and our approach is tracking-by-detection —
the detector is the lever, no track-id ground truth is needed. Hence plain
**image tasks** with detector pre-boxes. The interpolation-seed scripts were
removed to keep the branch to the one flow this guide describes.

---

## Decisions baked in (and why)

* **Detector pre-labelling, not from scratch.** Correcting boxes is ~5×
  faster than drawing them. The OFF price-tag detector is the same model
  the original auto-label notebook trusted; recall-first (`conf 0.05`) so
  you delete extras rather than miss tags.
* **Robot-parked de-dup.** The scan robot stops a lot; identical frames are
  wasted annotation effort and skew training. Dropped against the last
  *kept* frame so static runs collapse to one.
* **Image tasks, no attributes.** Detector-first; no OCR fields now or
  planned → no per-object details panel. Boxes only.
* **Detector-driven orientation, not fixed `ccw`.** Some clips are sideways,
  some are already upright, and some can be inverted. The detector now chooses
  orientation per scene before labels are written, so images and boxes always
  share the same geometry.
* **Photos use a separate raw root** (`data/raw_photos`): `detect_format`
  prefers CSV over YOLO under one root, so mixing would shadow a video CSV.
* **Pure, Unicode-safe adapters** (`ElementTree` + `cv_io`; cv2 only in CLI
  wrappers), unit-tested without OpenCV.

*Sources: [CVAT dataset formats](https://docs.cvat.ai/docs/dataset_management/formats/format-cvat/),
[import/export](https://docs.cvat.ai/docs/dataset_management/import-datasets/),
[installation](https://docs.cvat.ai/docs/administration/community/basics/installation/),
OpenFoodFacts [price-tag-detection](https://huggingface.co/openfoodfacts/price-tag-detection).*
