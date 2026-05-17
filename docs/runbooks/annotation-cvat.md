# Runbook — annotating & extending the dataset with CVAT

Hand-label new video / store photos (and clean the noisy released boxes —
briefing §6.1) in **CVAT 2.64.1**, then fold the result back into the
project with **zero pipeline change**: the CVAT export is converted into the
native 29-column Lenta CSV (video) or YOLO raw layout (photos), and the
existing `prepare_data.py` ingests it exactly like the original 5 videos.

> Why this is allowed: manual labelling **for training** is explicitly
> permitted (task.md §7) — it must be *disclosed* in the README (volume,
> method, how used). It is **forbidden at inference**. Keep that line.

CVAT lives in the repo as a git submodule at **`third_party/cvat`** (the
submodule gitlink pins it to commit `c0f002237`, CVAT 2.64.1). It is
brought up locally via its own Docker Compose — no
cloud, consistent with the inference constraint (task.md §10), though for
*labelling* a public dataset cloud would also be fine.

All commands assume cwd = this worktree
(`.claude/worktrees/annotation`) and the worktree venv
`.venv/Scripts/python.exe` (uv-managed; project rule: never global pip).

---

## 0. One-time: bring CVAT up

```bash
cd third_party/cvat
docker compose up -d                       # first run pulls images, ~minutes
# create your login:
docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
```

Open **http://localhost:8080** and log in. Leave it running — you do not
need to stop it between sessions, and stopping it does **not** lose data
(state is in Docker volumes, see §6).

To stop / resume later (data persists):

```bash
docker compose stop          # pause
docker compose up -d         # resume
# never `docker compose down -v` — the -v wipes the annotation volumes
```

---

## 1. One-time: create the project with the right label schema

The importer expects exactly one rectangle label `price_tag` with a fixed
attribute set (categorical `color`, `special_symbols`; the rest text). Emit
it and paste it in once:

```bash
.venv/Scripts/python.exe projects/price_tag_pipeline/scripts/cvat_prepare_task.py --label-spec
# writes cvat_label_spec.json and prints it + the full step list
```

In CVAT: **Projects → + → name "Lenta price tags" → open → Constructor →
toggle "Raw" → paste `cvat_label_spec.json` → Done.** Every task created
*inside this project* inherits the schema, so the export round-trips.

---

## 2. Create a task

### Video (this is where tracking works)

**Tasks → + →** Name = the **video_id** (e.g. `25_2-10` — the importer uses
it as the `filename` cell and CSV name), Project = "Lenta price tags",
upload the `.mp4`, submit. A video task is an **interpolation** task.

Released videos are 3840×2160 @ ~20 fps. Upload them **un-rotated** (see
§5 — rotating breaks coordinate compatibility with the released CSVs and
the training pipeline; the pipeline rotates uniformly downstream).

### Photos (store run — no tracking)

**Tasks → + →** Name = a **set id** (e.g. `store_run_1`), upload the images
as data, submit. Each photo is independent — there is no tracking and no
millisecond timestamp, so photos go down the YOLO/detection path (§4b).

---

## 3. Seed the existing annotations (video only)

So you *correct* the noisy released boxes instead of redrawing 274 of them:

```bash
# all 5 at once (reads the untouched real_data, needs the video for fps/size)
.venv/Scripts/python.exe projects/price_tag_pipeline/scripts/cvat_export_seed.py \
  --all --src "E:/Hackatons/lenta_tech_life_2026/real_data/dataset" --out cvat_seeds
# -> cvat_seeds/25_12-20.cvat.xml, 25_2-10.cvat.xml, ... (one <track> per tag)
```

In the task: **Actions → Upload annotations → format "CVAT 1.1" →** pick
`cvat_seeds/<video_id>.cvat.xml`. The released boxes appear as one track
per tag, with all known fields pre-filled as attributes. Fix the geometry
where it is off; leave the rest.

---

## 4a. Annotate a video — and how **tracking** works in CVAT

A video task is in **interpolation (track) mode**. The unit is a **Track**,
not a per-frame box:

- Draw a box on the first frame where a tag is clear and press **N** — this
  creates a Track and that frame becomes a **keyframe**.
- Scrub forward. The box is **linearly interpolated** between keyframes, so
  you only re-touch it where it drifts: move/resize it on a later frame and
  that frame auto-becomes another keyframe. Two keyframes are enough for a
  smooth pan; add more only where motion is non-linear.
- When the tag leaves the view, toggle **Outside** (the box stops counting
  from there) — keep the same Track; do not start a new one.
- Every Track has a stable **object id** shown on the box. **One physical
  price tag = one Track = one id, for its entire pass across the frame.**
  That id is the whole point: `cvat_import.py` writes it into a `track_id`
  column — free, exact ground truth for evaluating cross-track
  deduplication, which the metric punishes hardest (briefing §5.4/§6.5) and
  which the organizers explicitly leave to us.
- Optional speed-ups (not required): the **AI Tools → Tracker** (SiamMask /
  TransT) can auto-propagate a box across frames; **interpolation** alone is
  usually enough and fully deterministic. Either way the export is the same.
- The seed (§3) places each released tag as a **single-keyframe Track**
  bounded by an `outside` box one frame later, so it does not smear forward.
  Re-key it onto the frame where the tag is sharpest ("best frame",
  task.md §6.4) — that frame's timestamp is what gets scored.

**Per-tag attributes:** select the Track, set `color` / `special_symbols`
from the dropdown (fast). The text fields (name, prices, barcode, …) are
pre-filled from the seed for the 5 released videos — only fix them if the
seed is visibly wrong. For brand-new tags, filling text is optional:
geometry + `color`/`special_symbols` already extend the detector; barcode
and QR are decoded programmatically downstream, not typed.

## 4b. Annotate photos

Image task: just draw `price_tag` boxes (no tracks, no attributes needed —
detector training only). Set `color`/`special_symbols` if you want them in
future OCR training; otherwise boxes suffice.

---

## 5. Export

Task → **Actions → Export task dataset →**

- video → format **"CVAT for video 1.1"**
- photos → format **"CVAT for images 1.1"**

**"Save images" OFF** (we already have the source media). Download the
`.zip`, unzip → `annotations.xml`.

---

## 6. Hand it back — what I (Claude) run

You only deliver the XML (+ the photo folder, for photos). Then:

```bash
# VIDEO: XML -> data/raw/annotations/csv/<id>.csv (+ stages the video)
.venv/Scripts/python.exe projects/price_tag_pipeline/scripts/cvat_import.py \
  --xml <export>/annotations.xml \
  --video "E:/Hackatons/lenta_tech_life_2026/real_data/dataset/<id>/<id>.mp4"

# PHOTOS: XML -> data/raw_photos/{frames,annotations/labels}/<set>/  (YOLO)
.venv/Scripts/python.exe projects/price_tag_pipeline/scripts/cvat_import.py \
  --xml <export>/annotations.xml --images-dir <photo_folder> --set <set_id>

# then the unchanged pipeline:
.venv/Scripts/python.exe projects/price_tag_pipeline/scripts/prepare_data.py \
  --raw data/raw       --processed data/processed          # video
.venv/Scripts/python.exe projects/price_tag_pipeline/scripts/prepare_data.py \
  --raw data/raw_photos --processed data/processed_photos   # photos
```

Verified round-trip on `43_15` (real data): 29 tags in → 29 out, comma-
decimal bboxes parsed, Cyrillic names intact, `track_id` preserved, max
`frame_timestamp` drift **15 ms** vs a 50 ms frame — inside the matching
tolerance (task.md §6.3) and identical to the mapping the training pipeline
already applies, so new data behaves exactly like the original 5 videos.

---

## 7. Where everything is stored

| What | Location |
|---|---|
| CVAT source (pinned) | `third_party/cvat` (submodule @ commit `c0f002237`, CVAT 2.64.1) |
| **Your annotations + uploaded media** | Docker volume **`cvat_data`** → `/home/django/data` in `cvat_server` |
| CVAT relational state (tasks, users, labels) | Docker volume **`cvat_db`** (PostgreSQL `/var/lib/postgresql/data`) |
| CVAT keys / logs / events / caches | `cvat_keys`, `cvat_logs`, `cvat_events_db` (ClickHouse), `cvat_inmem_db` (Redis), `cvat_cache_db` (Kvrocks) |
| Seed XMLs (ours → CVAT) | `cvat_seeds/*.cvat.xml` (worktree, gitignored) |
| Label schema | `cvat_label_spec.json` (worktree) |
| Imported video annotations | `data/raw/annotations/csv/<id>.csv` (+ `videos/<id>.mp4`) |
| Imported photo annotations | `data/raw_photos/annotations/labels/<set>/` + `frames/<set>/` |
| Adapter code | `projects/price_tag_pipeline/src/price_tag_pipeline/data/cvat.py` + `scripts/cvat_*.py` |

Inspect a volume on the host: `docker volume inspect cvat_data`. To back up
all annotation state: stop CVAT and archive the `cvat_data` + `cvat_db`
volumes (`docker run --rm -v cvat_db:/v -v "$PWD:/b" busybox tar czf /b/cvat_db.tgz -C /v .`).

---

## 8. Decisions baked in (and why)

- **No 90° rotation by default.** Released CSV boxes are in the original,
  un-rotated frame; `ingest_lenta_csv` does not rotate; the pipeline rotates
  uniformly later. Rotating only the CVAT video would silently mis-place
  every seeded box. `cvat_prepare_task.py --rotate cw|ccw` exists but is
  off — turn it on only for a *fresh* set with no seed, knowing the whole
  set must then stay rotated.
- **Photos use a separate raw root** (`data/raw_photos`): `detect_format`
  prefers CSV over YOLO under one root, so a photo set there would shadow a
  video CSV. Different root = both ingest cleanly.
- **One Track → one CSV row**, using the largest visible keyframe as the
  best frame (task.md §6.4). Keeps new data homogeneous with the released
  one-row-per-tag CSVs and flows through the identical tested path.
- Adapters are pure (no cv2 in XML build/parse) and Unicode-safe
  (`ElementTree` + `cv_io`); only the CLI wrappers open videos. Covered by
  `tests/test_cvat.py` (runs without OpenCV, like `test_lenta_csv.py`).

---

*Sources: [CVAT docs — dataset formats](https://docs.cvat.ai/docs/dataset_management/formats/format-cvat/),
[import/export](https://docs.cvat.ai/docs/dataset_management/import-datasets/),
[installation](https://docs.cvat.ai/docs/administration/community/basics/installation/).*
