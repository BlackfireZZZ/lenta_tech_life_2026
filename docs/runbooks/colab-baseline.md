# ML Baseline (GPU) — Google Colab Runbook

Этот гайд покрывает только ML-часть: инференс, экспорт CSV под формат кейса, локальная оценка метрики.
Базовый путь ниже не требует разметки и обучения (zero-shot).

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

## 5. Batch inference по видео (zero-shot, без разметки)

```bash
!python projects/price_tag_pipeline/scripts/run_batch_inference.py \
  --videos-dir data/raw/videos \
  --config projects/price_tag_pipeline/configs/zeroshot_nolabel.yaml \
  --outputs-dir outputs/jsonl
```

## 6. Экспорт в CSV формата задания

```bash
!python projects/price_tag_pipeline/scripts/export_hack_csv.py \
  --inputs outputs/jsonl \
  --out-csv submission/hack_submission.csv \
  --video-ext .mp4
```

Итоговый файл: `submission/hack_submission.csv`.

## 7. Оценка end-to-end (если у вас есть GT CSV)

```bash
!python projects/price_tag_pipeline/scripts/eval_hack_csv.py \
  --pred-csv submission/hack_submission.csv \
  --gt-csv path/to/ground_truth.csv \
  --iou-thresh 0.3 \
  --tag-pass-threshold 0.8
```

## 8. Скачать артефакты из Colab

```bash
!zip -r artifacts.zip submission outputs/jsonl runs/lenta
```

После этого скачайте `artifacts.zip` из файлов Colab.

## 9. Опционально: усилить OCR (если есть VRAM)

Если на Colab A100 и хотите лучше качество текста, переключитесь на VLM-профиль:

```bash
!python projects/price_tag_pipeline/scripts/run_batch_inference.py \
  --videos-dir data/raw/videos \
  --config projects/price_tag_pipeline/configs/hq_glm_ocr.yaml \
  --outputs-dir outputs/jsonl
```
