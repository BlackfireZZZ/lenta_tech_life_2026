# Google Colab: private SSH clone, detector training, and full run

This runbook is written as copy-paste Colab cells for the private repository
`BlackfireZZZ/lenta_tech_life_2026`.

Primary goal: improve **price-tag detection** first. OCR quality is evaluated
only after the detector reliably finds the correct price tags and stops firing
on packages, bottles, shelf edges, and other objects.

Use a GPU runtime first:

`Runtime -> Change runtime type -> GPU`

Recommended runtime:

- **T4/L4**: smoke runs, YOLO11n/s training, quick visualization.
- **A100**: serious high-resolution YOLO training and later VLM/OCR work.

Expected Google Drive data layout from the current dataset screenshot:

```text
/content/drive/MyDrive/Fucking_нас/Данные/
├── 25_2-10/
│   ├── 25_2-10.mp4
│   └── 25_2-10.csv
├── 25_12-20/
├── 26_12-20/
├── 43_15/
├── 49_5/
├── Unlabeled/
│   ├── 25_12-20.mp4
│   ├── 26_12-20.mp4
│   └── 26_2-10.mp4
└── sample.csv
```

The current labeled set is tiny: **5 videos, about 63 annotated frames, 274
price-tag boxes**. Treat every experiment as data-centric: watch predictions,
fix/expand labels, then retrain.

## Cell 1 - generate SSH key in Colab

Use a temporary read-only deploy key. Regenerate it for each notebook/runtime.

```bash
%%bash
set -euo pipefail

mkdir -p ~/.ssh
chmod 700 ~/.ssh

if [ ! -f ~/.ssh/id_ed25519 ]; then
  ssh-keygen -t ed25519 \
    -C "colab-lenta-$(date +%Y%m%d-%H%M%S)" \
    -f ~/.ssh/id_ed25519 \
    -N ""
fi

ssh-keyscan github.com >> ~/.ssh/known_hosts
chmod 600 ~/.ssh/id_ed25519
chmod 644 ~/.ssh/known_hosts

echo "COPY THIS PUBLIC KEY TO GITHUB DEPLOY KEYS:"
cat ~/.ssh/id_ed25519.pub
```

Now add the printed public key to GitHub:

1. Open `https://github.com/BlackfireZZZ/lenta_tech_life_2026/settings/keys`.
2. Click `Add deploy key`.
3. Title: `colab-lenta`.
4. Key: paste the public key printed by Cell 1.
5. Keep write access disabled unless you explicitly need to push from Colab.
6. Click `Add key`.

If you cannot access repo settings, add the key to your GitHub account instead:
`https://github.com/settings/keys`.

## Cell 2 - test SSH access

GitHub usually exits with status `1` for `ssh -T` even when auth is successful,
so this cell intentionally allows that command to finish without failing the
notebook.

```bash
%%bash
set -euo pipefail
ssh -T git@github.com || true
```

Expected successful text contains:

```text
You've successfully authenticated
```

## Cell 3 - clone private repo and switch branch

```bash
%%bash
set -euo pipefail

cd /content

if [ ! -d lenta_tech_life_2026/.git ]; then
  git clone git@github.com:BlackfireZZZ/lenta_tech_life_2026.git
fi

cd /content/lenta_tech_life_2026
git fetch --all --prune
git switch feature/full-autonomous-demo
git pull --ff-only
git status --short --branch
```

## Cell 4 - install system packages and Python dependencies

```bash
%%bash
set -euo pipefail

cd /content/lenta_tech_life_2026

python -V
nvidia-smi || true

apt-get update
apt-get install -y libzbar0 tesseract-ocr tesseract-ocr-rus

python -m pip install -U pip setuptools wheel
python -m pip install -r projects/price_tag_pipeline/requirements/demo.txt
python -m pip install -r projects/price_tag_pipeline/requirements/train.txt
```

If Colab has dependency conflicts after repeated installs, restart the runtime
and rerun Cells 2-4.

## Cell 5 - smoke-check imports and CLI entry points

```bash
%%bash
set -euo pipefail

cd /content/lenta_tech_life_2026

python projects/price_tag_pipeline/scripts/prepare_data.py --help
python projects/price_tag_pipeline/scripts/make_splits.py --help
python projects/price_tag_pipeline/scripts/train_detector_yolo.py --help
python projects/price_tag_pipeline/scripts/eval_detector.py --help
python projects/price_tag_pipeline/scripts/run_batch_inference.py --help
python projects/price_tag_pipeline/scripts/visualize_predictions.py --help
```

## Cell 6 - mount Google Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

## Cell 7 - copy annotated CSV + videos into repo layout

Change `DATA_SRC` only if the Drive folder differs. This keeps labeled videos
separate from `Unlabeled`, because some file names can overlap.

```bash
%%bash
set -euo pipefail

cd /content/lenta_tech_life_2026

DATA_SRC="/content/drive/MyDrive/Fucking_нас/Данные"

mkdir -p data/raw/videos data/raw/annotations/csv data/raw/unlabeled_videos

# Labeled training/evaluation pairs: Данные/<video_id>/<video_id>.mp4 + .csv
find "${DATA_SRC}" -mindepth 2 -maxdepth 2 -type f -name "*.mp4" \
  ! -path "*/Unlabeled/*" \
  -exec cp -n {} data/raw/videos/ \;

find "${DATA_SRC}" -mindepth 2 -maxdepth 2 -type f -name "*.csv" \
  ! -name "sample.csv" \
  ! -path "*/Unlabeled/*" \
  -exec cp -n {} data/raw/annotations/csv/ \;

# Optional final/test videos without labels.
if [ -d "${DATA_SRC}/Unlabeled" ]; then
  cp -n "${DATA_SRC}/Unlabeled"/*.mp4 data/raw/unlabeled_videos/ || true
fi

echo "Labeled videos:"
find data/raw/videos -maxdepth 1 -type f | sort
echo "CSV annotations:"
find data/raw/annotations/csv -maxdepth 1 -type f | sort
echo "Unlabeled videos:"
find data/raw/unlabeled_videos -maxdepth 1 -type f | sort
```

## Cell 8 - inspect raw annotation volume

This is a quick reality check before training. Expect around 274 rows across
five CSV files. If counts are much lower, the Drive path or copy step is wrong.

```bash
%%bash
set -euo pipefail

cd /content/lenta_tech_life_2026

python - <<'PY'
from pathlib import Path
import csv

root = Path("data/raw/annotations/csv")
total = 0
for csv_path in sorted(root.glob("*.csv")):
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    total += len(rows)
    stamps = sorted({r.get("frame_timestamp", "") for r in rows})
    print(f"{csv_path.name}: rows={len(rows)} unique_timestamps={len(stamps)}")
print(f"total_rows={total}")
PY
```

## Cell 9 - prepare train-ready detector dataset from CSV

The CSV format contains `frame_timestamp` in **milliseconds** and bbox columns
`x_min,y_min,x_max,y_max`. This command extracts only annotated frames and
creates YOLO labels under `data/processed/`.

```bash
%%bash
set -euo pipefail

cd /content/lenta_tech_life_2026

python projects/price_tag_pipeline/scripts/prepare_data.py \
  --raw data/raw \
  --processed data/processed \
  --log-level INFO

python projects/price_tag_pipeline/scripts/make_splits.py \
  --processed data/processed \
  --out data/splits \
  --n_splits 5 \
  --emit-dataset-yaml-fold 0
```

## Cell 10 - train one fast detector smoke run

Use this to validate that training works end-to-end. This is not the final
model.

```bash
%%bash
set -euo pipefail

cd /content/lenta_tech_life_2026

python projects/price_tag_pipeline/scripts/train_detector_yolo.py \
  --dataset data/processed/dataset.yaml \
  --model yolo11n.pt \
  --imgsz 1280 \
  --batch 4 \
  --epochs 20 \
  --device 0 \
  --workers 2 \
  --name det_yolo11n_fold0_smoke
```

Evaluate the smoke checkpoint:

```bash
%%bash
set -euo pipefail

cd /content/lenta_tech_life_2026

python projects/price_tag_pipeline/scripts/eval_detector.py \
  --weights runs/lenta/det_yolo11n_fold0_smoke/weights/best.pt \
  --dataset data/processed/dataset.yaml \
  --imgsz 1280 \
  --batch 4 \
  --device 0
```

## Cell 11 - train the real 5-fold detector suite

Because there are only five labeled videos, each fold validates on one entire
video. This is the minimum honest setup; frame-level splits leak the same shelf
and the same tags into train and validation.

Start with the OpenFoodFacts price-tag detector. Use `yolo11s.pt` or
`yolo11m.pt` as ablation baselines; use `yolo11n.pt` only for speed. If YOLO26
weights are available in the current Ultralytics install, add a second suite
with `--model yolo26l.pt`.

```bash
%%bash
set -euo pipefail

cd /content/lenta_tech_life_2026

MODEL="yolo11s.pt"
IMGSZ=1280
BATCH=4
EPOCHS=160

for FOLD in 0 1 2 3 4; do
  python projects/price_tag_pipeline/scripts/make_splits.py \
    --processed data/processed \
    --out data/splits \
    --n_splits 5 \
    --emit-dataset-yaml-fold "${FOLD}"

  python projects/price_tag_pipeline/scripts/train_detector_yolo.py \
    --dataset data/processed/dataset.yaml \
    --model "${MODEL}" \
    --imgsz "${IMGSZ}" \
    --batch "${BATCH}" \
    --epochs "${EPOCHS}" \
    --device 0 \
    --workers 2 \
    --patience 40 \
    --use-albu \
    --name "det_${MODEL%.pt}_fold${FOLD}_img${IMGSZ}"

  python projects/price_tag_pipeline/scripts/eval_detector.py \
    --weights "runs/lenta/det_${MODEL%.pt}_fold${FOLD}_img${IMGSZ}/weights/best.pt" \
    --dataset data/processed/dataset.yaml \
    --imgsz "${IMGSZ}" \
    --batch "${BATCH}" \
    --device 0 | tee "runs/lenta/det_${MODEL%.pt}_fold${FOLD}_img${IMGSZ}_eval.txt"
done
```

## Cell 12 - choose and promote a checkpoint

Pick the checkpoint with the best validation behavior, not just the prettiest
loss curve. Production configs now default to the OpenFoodFacts detector; after
watching visualizations, copy the chosen local checkpoint and override
`detector.model_path` only if it beats that baseline.

```bash
%%bash
set -euo pipefail

cd /content/lenta_tech_life_2026

find runs/lenta -path "*/weights/best.pt" -print

# Change this after comparing eval logs and prediction videos.
BEST_RUN="runs/lenta/det_yolo11s_fold0_img1280/weights/best.pt"

mkdir -p data/checkpoints/detector
cp "${BEST_RUN}" data/checkpoints/detector/best.pt
ls -lh data/checkpoints/detector/best.pt
```

## Cell 13 - run inference on unlabeled videos with trained detector

Use `balanced.yaml` with either the default OpenFoodFacts detector or your
validated local checkpoint path.

```bash
%%bash
set -euo pipefail

cd /content/lenta_tech_life_2026
mkdir -p outputs/jsonl outputs/vis submission

VIDEOS_DIR="data/raw/unlabeled_videos"
CONFIG="projects/price_tag_pipeline/configs/balanced.yaml"

python projects/price_tag_pipeline/scripts/run_batch_inference.py \
  --videos-dir "${VIDEOS_DIR}" \
  --config "${CONFIG}" \
  --outputs-dir outputs/jsonl \
  --pattern "*.mp4" \
  --log-level INFO
```

## Cell 14 - render annotated videos for detector QA

Watch these videos first. Detector QA questions:

- Are true price tags missed?
- Are packages, bottles, boxes, shelf rails, or QR-like graphics detected?
- Are boxes tight enough to crop the whole tag for OCR?
- Are repeated detections merged by tracking, or do we double-count tags?

```bash
%%bash
set -euo pipefail

cd /content/lenta_tech_life_2026
mkdir -p outputs/vis

VIDEOS_DIR="data/raw/unlabeled_videos"
shopt -s nullglob

for v in "${VIDEOS_DIR}"/*.mp4; do
  stem="$(basename "$v" .mp4)"
  python projects/price_tag_pipeline/scripts/visualize_predictions.py \
    --video "$v" \
    --pred "outputs/jsonl/${stem}.jsonl" \
    --out "outputs/vis/${stem}_annotated.mp4" \
    --linger-frames 45
done
```

## Cell 15 - export hackathon CSV

```bash
%%bash
set -euo pipefail

cd /content/lenta_tech_life_2026

python projects/price_tag_pipeline/scripts/export_hack_csv.py \
  --inputs outputs/jsonl \
  --out-csv submission/hack_submission.csv \
  --video-ext .mp4

head -5 submission/hack_submission.csv
```

## Cell 16 - package outputs for download

```bash
%%bash
set -euo pipefail

cd /content/lenta_tech_life_2026
zip -r /content/lenta_results.zip submission outputs/jsonl outputs/vis runs/lenta data/splits
ls -lh /content/lenta_results.zip
```

Download `/content/lenta_results.zip` from the Colab file browser.

## Optional - run zero-shot baseline for comparison

This is useful only as a comparison point. It is expected to produce false
positives on packaging and other shelf objects.

```bash
%%bash
set -euo pipefail

cd /content/lenta_tech_life_2026
mkdir -p outputs/jsonl_zeroshot

python projects/price_tag_pipeline/scripts/run_batch_inference.py \
  --videos-dir data/raw/unlabeled_videos \
  --config projects/price_tag_pipeline/configs/zeroshot_nolabel.yaml \
  --outputs-dir outputs/jsonl_zeroshot \
  --pattern "*.mp4" \
  --log-level INFO
```

## Optional - fetch external detector datasets

External data should be train-only. Keep one real Lenta video as the validation
signal, otherwise external-domain quality will look better than the real task.

Show the available plan:

```bash
%%bash
set -euo pipefail

cd /content/lenta_tech_life_2026
python projects/price_tag_pipeline/scripts/fetch_external_datasets.py
```

Download SKU-110K for dense retail-shelf pretraining:

```bash
%%bash
set -euo pipefail

cd /content/lenta_tech_life_2026
python projects/price_tag_pipeline/scripts/fetch_external_datasets.py --download sku110k
```

Roboflow price-tag datasets require an API key:

```bash
%%bash
set -euo pipefail

cd /content/lenta_tech_life_2026

export ROBOFLOW_API_KEY="PASTE_KEY_HERE"
python projects/price_tag_pipeline/scripts/fetch_external_datasets.py \
  --download roboflow \
  --rf-workspace cuhk-00cw9 \
  --rf-project price-tag-mpq14 \
  --rf-version 1
```

After adding external datasets, normalize their classes to a single
`price_tag` class unless they explicitly separate useful price-tag subclasses.
Do not train product/package classes into the detector that feeds OCR; the OCR
pipeline needs price-tag crops, not product boxes.

## Optional - run Gradio UI in Colab

```bash
%%bash
set -euo pipefail

cd /content/lenta_tech_life_2026

python projects/price_tag_pipeline/scripts/gradio_app.py \
  --config projects/price_tag_pipeline/configs/balanced.yaml \
  --outputs-dir outputs/demo \
  --host 0.0.0.0 \
  --port 7860
```

## Optional - pull latest changes later

Use this when the notebook is already cloned and you only need to update it.

```bash
%%bash
set -euo pipefail

cd /content/lenta_tech_life_2026
git fetch --all --prune
git switch feature/full-autonomous-demo
git pull --ff-only
git log --oneline -5
```

## Optional - push from Colab

Only use this if the key has write access or was added to your GitHub account.
Do not commit input videos, checkpoints, or generated result archives.

```bash
%%bash
set -euo pipefail

cd /content/lenta_tech_life_2026

git config user.name "Colab Runner"
git config user.email "colab@example.invalid"

git status --short
# git add path/to/file
# git commit -m "Update Colab run output"
# git push origin HEAD:feature/full-autonomous-demo
```

## Detector experiment policy

For each detector run, save:

- model name and input size;
- fold id and validation video;
- `eval_detector.py` metrics;
- annotated videos from Cell 14;
- short notes: misses, false positives, bad crop tightness, duplicate tracks.

Promote a model only if it improves real-video visual QA. With the current tiny
dataset, mAP alone can lie.
