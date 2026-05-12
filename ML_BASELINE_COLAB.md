# ML Baseline (GPU) — Google Colab Runbook

Этот гайд покрывает только ML-часть: подготовка данных, обучение, инференс, экспорт CSV под формат кейса, локальная оценка метрики.

## 1. Создать Colab с GPU

1. Откройте новый notebook в Google Colab.
2. `Runtime -> Change runtime type -> T4/A100 GPU`.

## 2. Клонировать ветку с baseline

```bash
!git clone -b feature/ml-baseline-colab https://github.com/BlackfireZZZ/lenta_tech_life_2026.git
%cd lenta_tech_life_2026
```

## 3. Установить зависимости

```bash
!python -V
!pip install -U pip
!pip install -r projects/price_tag_pipeline/requirements.txt
```

Проверка GPU:

```bash
!nvidia-smi
```

## 4. Подготовить директории данных

```bash
!mkdir -p data/raw/videos data/raw/annotations data/processed data/splits data/checkpoints/detector outputs/jsonl submission
```

Дальше загрузите ваши видео в `data/raw/videos/`.

Если есть обучающая разметка, положите ее в `data/raw/annotations/` и при необходимости `data/raw/metadata.csv`.

## 5. Обучить baseline-детектор (если есть разметка)

```bash
!python projects/price_tag_pipeline/scripts/prepare_data.py \
  --raw data/raw \
  --processed data/processed

!python projects/price_tag_pipeline/scripts/make_splits.py \
  --processed data/processed \
  --metadata data/raw/metadata.csv \
  --out data/splits \
  --n_splits 5 \
  --emit-dataset-yaml-fold 0

!python projects/price_tag_pipeline/scripts/train_detector_yolo.py \
  --dataset data/processed/dataset.yaml \
  --model yolo26l.pt \
  --epochs 120 \
  --imgsz 1280 \
  --batch 8 \
  --device 0 \
  --name yolo26l_fold0_baseline
```

Скопировать лучший вес в дефолтный путь инференса:

```bash
!cp runs/lenta/yolo26l_fold0_baseline/weights/best.pt data/checkpoints/detector/best.pt
```

Оценка детектора:

```bash
!python projects/price_tag_pipeline/scripts/eval_detector.py \
  --weights data/checkpoints/detector/best.pt \
  --dataset data/processed/dataset.yaml
```

## 6. Batch inference по видео

```bash
!python projects/price_tag_pipeline/scripts/run_batch_inference.py \
  --videos-dir data/raw/videos \
  --config projects/price_tag_pipeline/configs/balanced.yaml \
  --outputs-dir outputs/jsonl
```

## 7. Экспорт в CSV формата задания

```bash
!python projects/price_tag_pipeline/scripts/export_hack_csv.py \
  --inputs outputs/jsonl \
  --out-csv submission/hack_submission.csv \
  --video-ext .mp4
```

Итоговый файл: `submission/hack_submission.csv`.

## 8. Оценка end-to-end (если у вас есть GT CSV)

```bash
!python projects/price_tag_pipeline/scripts/eval_hack_csv.py \
  --pred-csv submission/hack_submission.csv \
  --gt-csv path/to/ground_truth.csv \
  --iou-thresh 0.3 \
  --tag-pass-threshold 0.8
```

## 9. Скачать артефакты из Colab

```bash
!zip -r artifacts.zip submission outputs/jsonl runs/lenta
```

После этого скачайте `artifacts.zip` из файлов Colab.
