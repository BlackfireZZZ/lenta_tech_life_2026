# Data layout

The pipeline reads everything from this directory. **Once the dataset arrives, drop it into the matching subfolder and the scripts below will work without code changes.**

```
data/
├── raw/                          # staged dataset (local only — see "Git policy")
│   ├── videos/                   #   {video_id}.mp4  (labeled scenes)
│   ├── unlabeled_videos/         #   *.mp4  (no ground truth — inference/demo only)
│   └── annotations/
│       └── csv/                  #   {video_id}.csv  (+ sample.csv, submission ref)
├── processed/                    # outputs of prepare_data.py (frames + normalized labels)
│   ├── frames/                   #   {video_id}/{frame_idx:06d}.jpg
│   ├── labels/                   #   {video_id}/{frame_idx:06d}.txt   (YOLO format)
│   ├── gt_e2e/                   #   {video_id}.jsonl  (full-row E2E ground truth)
│   └── dataset.yaml              #   Ultralytics-compatible dataset YAML
├── splits/                       # video-level leave-one-out manifests (versioned in git)
│   └── fold_{0..4}.json
└── checkpoints/                  # trained weights (NOT in git — see .gitignore)
    ├── detector/
    └── vlm/
```

## The real dataset (current)

The organizer dataset is tiny — **5 labeled videos** + 3 unlabeled, ~600 MB:

| video_id   | rows | annotated frames |
|------------|-----:|-----------------:|
| 25_12-20   |   57 |               10 |
| 25_2-10    |   56 |                9 |
| 26_12-20   |   71 |               15 |
| 43_15      |   29 |                2 |
| 49_5       |   61 |               27 |

It ships in the organizer layout under `real_data/dataset/` (one folder per
video: `{id}/{id}.mp4` + `{id}/{id}.csv`, plus `sample.csv` and `Unlabeled/`).
`real_data/` is the untouched local source of truth and is **never committed**.
Project paths are kept ASCII by convention.

**Non-ASCII paths still work** as a robustness guarantee (the build machine
itself may sit under one — e.g. a Cyrillic Windows username
`C:\Users\<имя>\`). Still-image I/O goes through `price_tag_pipeline.cv_io`
(`cv2.imread/imwrite` silently fail on non-ASCII paths; cv2 *video* I/O is
fine), and the Ultralytics `processed/images` alias degrades symlink →
Windows junction → copy. So `--dst`/`--processed` may point anywhere.

Stage it into the canonical `data/raw/` layout (idempotent, non-destructive):

```bash
python projects/price_tag_pipeline/scripts/ingest_real_data.py \
  --src real_data/dataset --dst data/raw          # add --mode hardlink to save space
```

Because the set is this small, see [`../DATASETS.md`](../DATASETS.md) for
external price-tag / OCR datasets used to pre-train the detector.

## Git policy

`.gitignore` excludes all heavy data (`data/raw/videos/*`,
`data/raw/annotations/*`, `data/processed/*`, model checkpoints). Only the
small `data/splits/*.json` manifests and docs are versioned. After a fresh
clone or a new worktree, re-run `ingest_real_data.py` to repopulate `data/raw`
from your local `real_data/` — nothing fetches it for you.

## Expected file shapes

### Option A — Lenta hackathon CSV format

This is the actual annotated data format used by the current dataset. Put each
video in `data/raw/videos/` and the matching CSV in `data/raw/annotations/csv/`.
The CSV filename stem should match the video filename stem.

```
data/raw/
├── videos/
│   ├── 25_12-20.mp4
│   ├── 25_2-10.mp4
│   └── ...
└── annotations/
    └── csv/
        ├── 25_12-20.csv
        ├── 25_2-10.csv
        └── ...
```

Expected CSV columns include:

```
filename,product_name,price_default,price_card,...,frame_timestamp,x_min,y_min,x_max,y_max,...
```

`prepare_data.py` treats `frame_timestamp` as **milliseconds** (verified
against every clip's container duration — it is *not* a frame index), maps it
to a frame via the video's FPS, extracts only annotated frames, converts the
bbox columns into YOLO labels, and writes full-row ground truth to
`data/processed/gt_e2e/{video_id}.jsonl`. The header typo
`wholesale_level_1_coun` (present in `26_12-20.csv` / `43_15.csv`) is
normalized to `wholesale_level_1_count`.

### Option B — YOLO format

```
data/raw/
├── videos/
│   ├── store_07_aisle_03.mp4
│   └── ...
└── annotations/
    ├── classes.txt              # one class per line; expect a single "price_tag" line
    └── labels/
        └── store_07_aisle_03/
            ├── 000123.txt       # YOLO: class cx cy w h  (normalized 0..1)
            ├── 000124.txt
            └── ...
```

`classes.txt` example:
```
price_tag
```

### Option C — COCO format (auto-detected by extension `.json`)

```
data/raw/
├── videos/
│   └── *.mp4
└── annotations/
    └── instances.json           # COCO-format with image filenames mapping to {video_id}_{frame_idx:06d}.jpg
```

`prepare_data.py` auto-detects the format from the file structure.

### Option D — per-frame images already extracted

If the organizers ship frames instead of videos, drop them under `data/raw/frames/{video_id}/{frame_idx:06d}.jpg` and `prepare_data.py` will skip the frame extraction step.

## How to use this directory

```bash
# 1. Stage the raw dataset into data/raw/ (organizer layout -> canonical layout).
python projects/price_tag_pipeline/scripts/ingest_real_data.py --src real_data/dataset --dst data/raw

# 2. Validate integrity, extract frames if needed, build the unified YOLO dataset.
python projects/price_tag_pipeline/scripts/prepare_data.py \
  --raw data/raw \
  --processed data/processed

# 3. Build video-level 5-fold split.
python projects/price_tag_pipeline/scripts/make_splits.py \
  --processed data/processed \
  --out data/splits \
  --n_splits 5

# 4. Train the detector against fold 0.
python projects/price_tag_pipeline/scripts/train_detector_yolo.py \
  --dataset data/processed/dataset.yaml \
  --fold 0
```

## Notes

- Per-video metadata (store id, aisle, camera, time-of-day) is optional. If you have it,
  add `data/raw/metadata.csv` with columns `video_id,store_id,aisle_id,camera_id,recorded_at`
  and the split script will stratify by `(store_id, aisle_id)` automatically.
- All paths in pipeline configs are relative to the repo root.
- Frames are JPEG-encoded at 95% quality unless `--frame-quality` is overridden.
- `data/processed/` is rebuilt by `prepare_data.py`; safe to delete and regenerate.

## Without the dataset

`data/raw/` and `data/processed/` are git-empty (only `.gitkeep`). Pure data
parsing (CSV/labels/splits) imports and tests without OpenCV; frame extraction
and `validate_dataset(read_images=True)` need `opencv-python`. Unit tests pass
and smoke tests run against synthetic fixtures.
