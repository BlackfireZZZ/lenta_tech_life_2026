# Google Colab: private SSH clone and full run

This runbook is written as copy-paste Colab cells for the private repository
`BlackfireZZZ/lenta_tech_life_2026`.

Use a GPU runtime first:

`Runtime -> Change runtime type -> GPU`

Recommended path for a private repo is an ephemeral read-only SSH deploy key:
generate it inside Colab, add the printed public key to GitHub, clone, install,
copy videos, run inference, export CSV, package results.

## Cell 1 — generate SSH key in Colab

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

## Cell 2 — test SSH access

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

## Cell 3 — clone private repo and switch branch

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

## Cell 4 — install system packages and Python dependencies

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

## Cell 5 — smoke-check imports and CLI entry points

```bash
%%bash
set -euo pipefail

cd /content/lenta_tech_life_2026

python projects/price_tag_pipeline/scripts/run_batch_inference.py --help
python projects/price_tag_pipeline/scripts/export_hack_csv.py --help
python projects/price_tag_pipeline/scripts/gradio_app.py --help
```

## Cell 6 — mount Google Drive

Use this if the input videos are stored on Google Drive.

```python
from google.colab import drive
drive.mount("/content/drive")
```

## Cell 7 — copy annotated CSV + videos into repo layout

Change `DATA_SRC` to the folder where the downloaded `Данные` directory is
located. This keeps labeled videos separate from `Unlabeled`, because some file
names can overlap.

```bash
%%bash
set -euo pipefail

cd /content/lenta_tech_life_2026

DATA_SRC="/content/drive/MyDrive/Данные"

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

## Cell 8 — prepare train-ready detector dataset from CSV

The CSV format contains `frame_timestamp` and bbox columns
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

## Cell 9 — optional detector training

Use this if you want to train on the provided CSV boxes. For a quick Colab smoke
run, lower `--epochs`; for a real run, increase it.

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
  --name lenta_csv_smoke
```

After training, copy the best checkpoint into the default config path:

```bash
%%bash
set -euo pipefail

cd /content/lenta_tech_life_2026
mkdir -p data/checkpoints/detector
cp runs/lenta/lenta_csv_smoke/weights/best.pt data/checkpoints/detector/best.pt
```

## Cell 10 — run inference on unlabeled videos

For a no-training baseline, keep `zeroshot_nolabel.yaml`. If you trained and
copied `best.pt`, use `balanced.yaml`.

```bash
%%bash
set -euo pipefail

cd /content/lenta_tech_life_2026
mkdir -p outputs/jsonl outputs/vis submission

VIDEOS_DIR="data/raw/unlabeled_videos"
CONFIG="projects/price_tag_pipeline/configs/zeroshot_nolabel.yaml"

python projects/price_tag_pipeline/scripts/run_batch_inference.py \
  --videos-dir "${VIDEOS_DIR}" \
  --config "${CONFIG}" \
  --outputs-dir outputs/jsonl \
  --pattern "*.mp4" \
  --log-level INFO
```

## Cell 11 — render annotated videos

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

## Cell 12 — export hackathon CSV

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

## Cell 13 — package outputs for download

```bash
%%bash
set -euo pipefail

cd /content/lenta_tech_life_2026
zip -r /content/lenta_results.zip submission outputs/jsonl outputs/vis
ls -lh /content/lenta_results.zip
```

Download `/content/lenta_results.zip` from the Colab file browser.

## Optional cell — run Gradio UI in Colab

```bash
%%bash
set -euo pipefail

cd /content/lenta_tech_life_2026

python projects/price_tag_pipeline/scripts/gradio_app.py \
  --config projects/price_tag_pipeline/configs/zeroshot_nolabel.yaml \
  --outputs-dir outputs/demo \
  --host 0.0.0.0 \
  --port 7860
```

## Optional cell — pull latest changes later

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

## Optional cell — push from Colab

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
