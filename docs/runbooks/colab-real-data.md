# Google Colab Runbook for Real Videos

Use this when the repository is private and you need to run the full autonomous
pipeline on the three public videos.

## 1. Start Colab

1. Create a new Google Colab notebook.
2. Runtime -> Change runtime type -> GPU.
3. Prefer A100 for VLM profiles; T4 is enough for zero-shot + PaddleOCR.

## 2. Clone Private Repo over SSH

Generate or paste a deploy key with read access to the private repo.

```bash
!mkdir -p ~/.ssh
```

In a Colab cell, paste your private key into `/root/.ssh/id_ed25519`.
Do not commit this key anywhere.

```bash
%%bash
cat > ~/.ssh/id_ed25519 <<'EOF'
PASTE_PRIVATE_KEY_HERE
EOF
chmod 600 ~/.ssh/id_ed25519
ssh-keyscan github.com >> ~/.ssh/known_hosts
```

Clone and switch branch:

```bash
!git clone git@github.com:BlackfireZZZ/lenta_tech_life_2026.git
%cd lenta_tech_life_2026
# main is the canonical branch
```

If your key has a passphrase, use Colab secrets or `ssh-agent`; for hackathon
speed, a temporary read-only deploy key without passphrase is usually simpler.

## 3. Install Dependencies

Zero-shot + classical OCR:

```bash
!python -V
!pip install -U pip
!pip install -r projects/price_tag_pipeline/requirements/demo.txt
!nvidia-smi
```

If Paddle GPU wheels are needed, install the correct `paddlepaddle-gpu` build
for the Colab CUDA version before `paddleocr`.

## 4. Upload Videos

Option A: upload manually through the Colab file browser.

Option B: mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Then:

```bash
!mkdir -p data/raw/videos outputs/jsonl outputs/vis submission
!cp /content/drive/MyDrive/lenta_public_videos/*.mp4 data/raw/videos/
```

## 5. Run No-Label Baseline

```bash
!python projects/price_tag_pipeline/scripts/run_batch_inference.py \
  --videos-dir data/raw/videos \
  --config projects/price_tag_pipeline/configs/zeroshot_nolabel.yaml \
  --outputs-dir outputs/jsonl \
  --log-level INFO
```

## 6. Render Annotated Videos

```bash
!for v in data/raw/videos/*.mp4; do \
  stem=$(basename "$v" .mp4); \
  python projects/price_tag_pipeline/scripts/visualize_predictions.py \
    --video "$v" \
    --pred "outputs/jsonl/${stem}.jsonl" \
    --out "outputs/vis/${stem}_annotated.mp4" \
    --linger-frames 45; \
done
```

If you used a config with `runtime.audit_path`, add `--audit outputs/audit.jsonl`
for the single-video command to see OCR/QR moments in the top-left panel.

## 7. Export Final CSV

```bash
!python projects/price_tag_pipeline/scripts/export_hack_csv.py \
  --inputs outputs/jsonl \
  --out-csv submission/hack_submission.csv \
  --video-ext .mp4
```

## 8. Package Results

```bash
!zip -r lenta_results.zip submission outputs/jsonl outputs/vis
```

Download `lenta_results.zip` from the Colab file browser.

## 9. Stronger Run for Better Score

If GPU memory allows, run a VLM profile:

```bash
!python projects/price_tag_pipeline/scripts/run_batch_inference.py \
  --videos-dir data/raw/videos \
  --config projects/price_tag_pipeline/configs/hq_ensemble.yaml \
  --outputs-dir outputs/jsonl_hq \
  --log-level INFO
```

Then export:

```bash
!python projects/price_tag_pipeline/scripts/export_hack_csv.py \
  --inputs outputs/jsonl_hq \
  --out-csv submission/hack_submission_hq.csv \
  --video-ext .mp4
```

## 10. What To Tune After Watching Annotated Videos

If many tags are missed:

```yaml
detector:
  conf: 0.12
  open_vocab_labels:
    - "price tag"
    - "shelf price label"
    - "yellow supermarket price label"
    - "barcode price label"
```

If too many false boxes appear:

```yaml
detector:
  conf: 0.25
aggregation:
  min_final_confidence: 0.45
```

If OCR is weak:

```yaml
ocr:
  min_sharpness: 15.0
  top_k_crops_per_track: 8
rectifier:
  super_resolution: true
```

If duplicates survive:

```yaml
aggregation:
  dedup_iou_threshold: 0.25
  dedup_time_window_frames: 260
```
