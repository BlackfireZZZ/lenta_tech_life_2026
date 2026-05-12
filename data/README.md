# Data layout

The pipeline reads everything from this directory. **Once the dataset arrives, drop it into the matching subfolder and the scripts below will work without code changes.**

```
data/
├── raw/                          # untouched dataset from organizers
│   ├── videos/                   #   put .mp4 / .avi / .mov files here, one per scene
│   └── annotations/              #   YOLO labels (.txt) + classes.txt, OR COCO json
├── processed/                    # outputs of prepare_data.py (frames + normalized labels)
│   ├── frames/                   #   {video_id}/{frame_idx:06d}.jpg
│   ├── labels/                   #   {video_id}/{frame_idx:06d}.txt   (YOLO format)
│   └── dataset.yaml              #   Ultralytics-compatible dataset YAML
├── splits/                       # video-level GroupKFold manifests (versioned in git)
│   └── fold_{0..4}.json
└── checkpoints/                  # trained weights (NOT in git — see .gitignore)
    ├── detector/
    └── vlm/
```

## Expected file shapes

### Option A — YOLO format (assumed default)

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

### Option B — COCO format (auto-detected by extension `.json`)

```
data/raw/
├── videos/
│   └── *.mp4
└── annotations/
    └── instances.json           # COCO-format with image filenames mapping to {video_id}_{frame_idx:06d}.jpg
```

`prepare_data.py` auto-detects the format from the file structure.

### Option C — per-frame images already extracted

If the organizers ship frames instead of videos, drop them under `data/raw/frames/{video_id}/{frame_idx:06d}.jpg` and `prepare_data.py` will skip the frame extraction step.

## How to use this directory

```bash
# 1. Put the raw dataset under data/raw/ as described above.

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

## Pre-data state

Until data arrives, `data/raw/` and `data/processed/` are empty (only `.gitkeep`).
The pipeline modules will still import, all unit tests will pass, and the smoke
tests will run against synthetic fixtures under `tests/fixtures/`.
