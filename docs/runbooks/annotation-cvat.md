# Guide — annotating & extending the dataset with CVAT

How to use **our** CVAT, push annotations into it, run every adapter tool,
and fold the result back into the project with **zero pipeline change**: a
CVAT export becomes either the native 29‑column Lenta CSV (video, fields
filled) or the YOLO raw layout (boxes‑only / photos), and the existing
`prepare_data.py` ingests it exactly like the original 5 videos.

> **Allowed?** Manual labelling **for training** is explicitly permitted
> (task.md §7) and must be *disclosed* in the README (volume, method, how
> used). It is **forbidden at inference**. Keep that line.

CVAT is vendored as a git submodule at **`third_party/cvat`** (gitlink‑pinned
to commit `c0f002237`, CVAT 2.64.1) and runs locally via its own Docker
Compose — no cloud, consistent with the inference constraint (task.md §10).

All commands assume **cwd = this worktree** (`.claude/worktrees/annotation`)
and the worktree venv **`.venv/Scripts/python.exe`** (uv‑managed; project
rule: never global pip). Replace `***` with the admin password you set.

---

## TL;DR

1. `cd third_party/cvat && docker compose up -d` → open http://localhost:8080.
2. Open a task in the **"Lenta price tags"** project, **draw boxes** on
   price tags (label `price_tag`, no fields — boxes only).
3. **Actions → Export task dataset →** "CVAT for video 1.1" (video) or
   "CVAT for images 1.1" (photos), *Save images OFF*.
4. Hand the `annotations.xml` back → `cvat_import.py` + `prepare_data.py`
   extend the dataset.

The 5 released videos and the friend's 4 photo scenes are **already loaded
with seed boxes** — you mostly validate/extend, not start from scratch.

---

## 1. Setup (one‑time)

```bash
cd third_party/cvat
docker compose up -d                       # first run pulls images, ~minutes
docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
```

Open http://localhost:8080 and log in. The project **"Lenta price tags"**
holds one **bare** rectangle label `price_tag` (no attributes → no
per‑object "details" panel; detector‑first, boxes only). Pause/resume any
time — state lives in Docker volumes (§7):

```bash
docker compose stop      # pause      |   docker compose up -d   # resume
# NEVER `docker compose down -v`  — the -v wipes all annotation volumes
```

Submodule clone of CVAT failing on flaky network? Shallow + pinned:
`git -c http.version=HTTP/1.1 clone --depth 1 --branch v2.64.0 \
https://github.com/cvat-ai/cvat third_party/cvat` then
`git submodule add --force https://github.com/cvat-ai/cvat third_party/cvat`.

---

## 2. Concepts

| | **Video task** | **Image task (photos)** |
|---|---|---|
| Mode | interpolation / **tracking** | independent frames |
| Unit | a **Track** (stable id across frames) | a box per image |
| Use | released videos, robot clips | store photos, friend datasets |
| Import path | `tags` (fields) or `detector` (boxes) | always `detector` (YOLO) |

**Detector‑first.** We only draw boxes; OCR fields are not annotated (now
or planned). Two import paths, auto‑selected by `cvat_import.py`:

- **boxes only → `detector`**: tracks interpolated between keyframes →
  dense per‑frame YOLO under `data/raw_det`; photos → YOLO under
  `data/raw_photos`. The cheap way to grow the detector.
- **fields filled → `tags`**: one Track → one 29‑col Lenta CSV row
  (`frame_timestamp` in ms, `track_id` carried as cross‑track‑dedup GT)
  under `data/raw`. Only if you ever do OCR ground truth.

Both feed the **unchanged** `prepare_data.py`.

---

## 3. Tool reference

All under `projects/price_tag_pipeline/scripts/`, run with the venv python.
Pure XML build/parse is cv2‑free and unit‑tested (`tests/test_cvat.py`,
`data/cvat.py`).

| Tool | Purpose | Key flags | Example |
|---|---|---|---|
| `cvat_prepare_task.py` | Emit the bare `price_tag` label spec + print task‑setup steps | `--label-spec`, `--rotate cw\|ccw` (off by default) | `… cvat_prepare_task.py --label-spec` |
| `cvat_export_seed.py` | Released Lenta CSV → **CVAT‑video seed XML** (one track per tag) so you correct, not redraw | `--all --src <real_data>`, or `--csv --video`, `--out` | `… cvat_export_seed.py --all --src "E:/…/real_data/dataset" --out cvat_seeds` |
| `cvat_bootstrap.py` | Create project + one **video task per released video**, upload mp4, import its seed (cvat‑sdk) | `--src`, `--seeds`, `--user/--password`, `--only` | `… cvat_bootstrap.py --src "E:/…/real_data/dataset" --seeds cvat_seeds --user admin --password ***` |
| `cvat_from_external.py` | Friend dump (candidates CSV **or** classic YOLO‑txt) → N **chronological scene** image‑XMLs + `manifest.json` | `--csv` / `--yolo-labels`, `--model-backed`, `--sources`, `--scenes`, `--min-conf` | `… cvat_from_external.py --csv friends_labels/all_candidates2.csv --model-backed --images-dir friends_labels/dataset_lenta --scenes 4 --out cvat_seeds_friends` |
| `cvat_bootstrap_photos.py` | Create one **image task per scene** from a manifest, upload photos, import seeds | `--manifest`, `--user/--password`, `--replace`, `--only` | `… cvat_bootstrap_photos.py --manifest cvat_seeds_friends/manifest.json --user admin --password ***` |
| `cvat_import.py` | **CVAT export → ours.** Auto: boxes‑only→detector, fields→tags; photos→YOLO | `--xml`, `--video` \| `--images-dir --set`, `--mode auto\|tags\|detector` | `… cvat_import.py --xml ann.xml --images-dir friends_labels/dataset_lenta --set friends_scene_2` |
| `cvat_strip_attributes.py` | Remove all attributes from the **live** project label (kills the details panel; no re‑upload) | `--user/--password`, `--project` | `… cvat_strip_attributes.py --user admin --password ***` |
| `prepare_data.py` | Existing pipeline ingest — **unchanged** | `--raw <root> --processed <out>` | `… prepare_data.py --raw data/raw_photos --processed data/processed_photos` |

---

## 4. Workflows

### A. Released videos (5 clips, seed already loaded)

Tasks `25_12-20 … 49_5` exist with one track per released tag. Open a task:

1. Fix obviously wrong seed boxes; **draw boxes on the many missing tags**.
2. Per tag, set **2 keyframes** (first clear frame + last before it leaves);
   CVAT interpolates the box on every frame between → dozens of labelled
   detector frames for two clicks. Toggle **Outside** when it leaves; keep
   the **same Track**. Don't touch fields.
3. **Export** → "CVAT for video 1.1", *Save images OFF*.
4. Hand back → I run:

```bash
.venv/Scripts/python.exe projects/price_tag_pipeline/scripts/cvat_import.py \
  --xml <export>/annotations.xml \
  --video "E:/Hackatons/lenta_tech_life_2026/real_data/dataset/<id>/<id>.mp4"
# auto: boxes-only → data/raw_det ; fields → data/raw  (force with --mode)
.venv/Scripts/python.exe projects/price_tag_pipeline/scripts/prepare_data.py \
  --raw data/raw_det --processed data/processed_det
```

(Re)generate seeds yourself with `cvat_export_seed.py --all …`; load via the
task's **Actions → Upload annotations → "CVAT 1.1"**.

### B. Your own store photos

`cvat_prepare_task.py --label-spec` → create an **image task** (name = a set
id), upload photos, draw `price_tag` boxes, **Export "CVAT for images 1.1"**.
Then:

```bash
.venv/Scripts/python.exe projects/price_tag_pipeline/scripts/cvat_import.py \
  --xml <export>/annotations.xml --images-dir <photo_folder> --set <set_id>
.venv/Scripts/python.exe projects/price_tag_pipeline/scripts/prepare_data.py \
  --raw data/raw_photos --processed data/processed_photos
```

Shoot guidance: 4K, landscape, **don't pre‑rotate**, H.264 CFR / JPEG,
ASCII folder per zone, focus on barcode + hard cases (glass, glare, blur).

### C. Friend / external dataset (already loaded — `friends_scene_1..4`)

`friends_labels/` ships a candidates CSV. `all_candidates2.csv` is the full
run over all **132** mixed portrait/landscape photos in
`friends_labels/dataset_lenta/`. `merged_final` is recall‑first
pseudo‑labelling (YOLO `conf≥0.05` + a colour‑CV heuristic); **`--model-backed`**
drops the noisy `classic_color_cv` (~⅔) → ~**1958** YOLO‑backed boxes,
split into 4 chronological scenes (≈490 each, all 132 photos boxed).

```bash
# regenerate scenes (already done once)
.venv/Scripts/python.exe projects/price_tag_pipeline/scripts/cvat_from_external.py \
  --csv friends_labels/all_candidates2.csv --model-backed \
  --images-dir friends_labels/dataset_lenta --scenes 4 --out cvat_seeds_friends

# (re)load into CVAT — --replace re-imports WITHOUT re-uploading photos
.venv/Scripts/python.exe projects/price_tag_pipeline/scripts/cvat_bootstrap_photos.py \
  --manifest cvat_seeds_friends/manifest.json --user admin --password *** [--replace]
```

Validate/fix boxes in `friends_scene_1..4`, then per scene:

```bash
.venv/Scripts/python.exe projects/price_tag_pipeline/scripts/cvat_import.py \
  --xml <export>/annotations.xml \
  --images-dir friends_labels/dataset_lenta --set friends_scene_2
.venv/Scripts/python.exe projects/price_tag_pipeline/scripts/prepare_data.py \
  --raw data/raw_photos --processed data/processed_photos
```

(Classic YOLO‑txt dump instead of a CSV? swap `--csv` for `--yolo-labels <dir>`.)

---

## 5. How tracking works in CVAT (video tasks)

A video task is **interpolation mode**; the unit is a **Track**, not a
per‑frame box:

- Draw a box, press **N** → a Track is created, that frame is a **keyframe**.
- Move/resize it on a later frame → that frame auto‑becomes a keyframe;
  CVAT **linearly interpolates** the box on every frame between. Two
  keyframes cover a smooth pan; add more only for non‑linear motion.
- Toggle **Outside** when the tag leaves view — same Track, don't start a
  new one.
- Each Track has a stable **id** = one physical tag across its whole pass.
  `cvat_import.py` writes it as a `track_id` column → free, exact ground
  truth for cross‑track dedup (the metric's hardest part, briefing §5.4/§6.5).
- AI Tools → Tracker (SiamMask/TransT) can auto‑propagate; interpolation
  alone is deterministic and enough. Export is identical either way.

---

## 6. Export from CVAT

Task → **Actions → Export task dataset →** format **"CVAT for video 1.1"**
(video) or **"CVAT for images 1.1"** (photos), **"Save images" OFF** (we
already have the media). Unzip → `annotations.xml` → hand back.

Round‑trip verified on `43_15`: 29 tags in → 29 out, comma‑decimal bboxes
parsed, Cyrillic names intact, `track_id` preserved, max `frame_timestamp`
drift **15 ms** vs a 50 ms frame — inside the matching tolerance (task.md
§6.3) and identical to the mapping the training pipeline already applies.

---

## 7. Where everything is stored

| What | Location |
|---|---|
| CVAT source (pinned) | `third_party/cvat` (submodule @ `c0f002237`, 2.64.1) |
| **Annotations + uploaded media** | Docker volume **`cvat_data`** → `/home/django/data` |
| Tasks / users / labels | Docker volume **`cvat_db`** (PostgreSQL) |
| Keys / logs / events / caches | `cvat_keys`, `cvat_logs`, `cvat_events_db`, `cvat_inmem_db`, `cvat_cache_db` |
| Released‑video seeds | `cvat_seeds/*.cvat.xml` (gitignored) |
| Friend scene seeds + manifest | `cvat_seeds_friends/` (gitignored) |
| Friend raw dataset | `friends_labels/` (gitignored: photos, `all_candidates*.csv`) |
| Imported video — tags / detector | `data/raw/annotations/csv/<id>.csv` / `data/raw_det/…` |
| Imported photos | `data/raw_photos/{frames,annotations/labels}/<set>/` |
| Adapter code | `src/price_tag_pipeline/data/cvat.py` + `scripts/cvat_*.py` |

Inspect: `docker volume inspect cvat_data`. Backup (stop CVAT first):
`docker run --rm -v cvat_db:/v -v "$PWD:/b" busybox tar czf /b/cvat_db.tgz -C /v .`
(repeat for `cvat_data`).

---

## 8. Troubleshooting

- **"I don't see any boxes."** You opened the wrong task/frame, or the
  project/task list (boxes only show inside a **Job**). Released `scene_1`
  style seeds may be sparse on early frames. Friend scenes: every photo is
  boxed — open `friends_scene_2`'s job. Check label **Opacity** in the
  right panel; 4K images make ~400 px boxes look tiny at fit‑to‑screen.
- **"The per‑box details panel annoys me."** It's label attributes. Run
  `cvat_strip_attributes.py --user admin --password ***` — strips them off
  the live label (no re‑upload, boxes untouched). New projects are already
  bare.
- **Re‑filtered the friend CSV, don't want to re‑upload 132 photos.**
  Regenerate scenes, then `cvat_bootstrap_photos.py … --replace` (clears +
  re‑imports annotations only).
- **CVAT won't start / port busy.** `docker compose ps`; logs:
  `docker compose logs cvat_server`. First boot runs DB migrations (~1 min).
- **Submodule clone fails (RPC reset).** Shallow + pinned (see §1).
- **`cvat-sdk` missing.** `uv pip install --python .venv cvat-sdk`.

---

## 9. Decisions baked in (and why)

- **Bare `price_tag` label, no attributes.** Detector‑first; no OCR fields
  now or planned → no details panel. The parser still reads `<attribute>`
  if a future XML carries them.
- **No 90° rotation by default.** Released CSV boxes are un‑rotated;
  `ingest_lenta_csv` doesn't rotate; the pipeline rotates uniformly later.
  Rotating only the CVAT video would silently mis‑place every seed.
  `cvat_prepare_task.py --rotate` exists for *fresh* sets with no seed.
- **Photos use a separate raw root** (`data/raw_photos`/`data/raw_det`):
  `detect_format` prefers CSV over YOLO under one root, so mixing would
  shadow a video CSV.
- **Friend `merged_final` filtered to `--model-backed`.** The friend's own
  notebook flags `classic_color_cv` as low‑trust; dropping it removes ~⅔
  noise while keeping YOLO recall.
- **Pure, Unicode‑safe adapters** (`ElementTree` + `cv_io`; cv2 only in CLI
  wrappers), tested without OpenCV like `test_lenta_csv.py`.

---

*Sources: [CVAT dataset formats](https://docs.cvat.ai/docs/dataset_management/formats/format-cvat/),
[import/export](https://docs.cvat.ai/docs/dataset_management/import-datasets/),
[installation](https://docs.cvat.ai/docs/administration/community/basics/installation/).*
