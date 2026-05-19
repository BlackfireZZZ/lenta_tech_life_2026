# Full data upload: Drive -> Colab -> CVAT -> training

Эта инструкция для ветки `worktree-annotation`, когда нужно догрузить все
данные в разметочный контур: новые видео, дополнительные фото для ручной
дораметки и новые предобученные веса детектора.

Основной путь:

1. Сложить данные в Google Drive в понятную структуру.
2. Запустить Colab notebook
   `notebooks/lenta_full_data_cvat_colab.ipynb`.
3. Скачать `cvat_video_frames_prelabelled_full.zip`.
4. На ноутбуке импортировать кадры + предразметку в CVAT.
5. В CVAT поправить боксы.
6. Выгрузить обратно в `data/raw_photos`, собрать `data/processed_photos`,
   дообучить детектор.

## 0. Что сейчас важно

Рабочая ветка с CVAT-потоком:

```bash
cd /Users/cute/Lenta_Tech/ultralytics-annotation
git status --short --branch
```

Ожидаемо должно быть `worktree-annotation`. В этой ветке уже есть:

- `docs/runbooks/annotation-cvat.md` - базовый CVAT-пайплайн.
- `notebooks/lenta_prelabel_colab.ipynb` - старый Colab только для уже
  нарезанных кадров.
- `notebooks/lenta_full_data_cvat_colab.ipynb` - новый Colab для полного
  сценария: видео + фото + кастомные веса.

## 1. Как загрузить данные в Google Drive

Создай в Drive папку:

```text
MyDrive/lenta_full_data/
```

Внутри держи такую структуру:

```text
lenta_full_data/
  videos/
    25_12-20/25_12-20.mp4
    26_12-20/26_12-20.mp4
    new_store_01.mp4
    new_store_02.mp4
  photos/
    glare_cases/*.jpg
    alcohol_shelf/*.jpg
    blur_or_motion/*.jpg
    hard_negatives/*.jpg
  weights/
    best.pt
```

Правила:

- `videos/` можно делать рекурсивной: каждый `.mp4/.mov/.mkv/.avi` будет
  найден. Имя CVAT task строится из папки + имени файла.
- Если видео от робота такие же, как текущие, они лежат боком. Notebook
  повернет кадры `ccw` по умолчанию, чтобы ценники были вертикально.
- `photos/` - любые дополнительные фото для доразметки. Верхняя папка
  станет группой CVAT task, например `photos_extra__glare_cases`.
- `weights/best.pt` - новые веса с диска. Если файла нет, notebook сам
  возьмет OpenFoodFacts `price-tag-detection/weights/best.pt`.
- Не смешивай исходные видео и сгенерированные кадры вручную. Notebook сам
  создает `lenta_full_data/cvat_video_frames/`.

## 2. Запуск нового Colab notebook

Открой Colab:

```text
https://colab.research.google.com
```

Дальше:

1. `File -> Upload notebook`.
2. Выбери
   `notebooks/lenta_full_data_cvat_colab.ipynb`.
3. `Runtime -> Change runtime type -> T4 GPU`.
4. Запусти все ячейки сверху вниз.

Что делает notebook:

- монтирует Google Drive;
- ищет видео в `MyDrive/lenta_full_data/videos`;
- нарезает кадры каждые `2.0` секунды;
- убирает почти одинаковые кадры, когда робот стоит;
- автоматически подбирает ориентацию каждой видео-сцены перед предразметкой:
  пробует `none`, `cw`, `ccw`, `180`, выбирает вариант с лучшим detector
  score, физически поворачивает изображения и чистит старые `.txt`, если
  геометрия поменялась;
- добавляет фото из `MyDrive/lenta_full_data/photos`;
- берет веса из `MyDrive/lenta_full_data/weights/best.pt`, если они есть;
- прогоняет YOLO с recall-first настройками:
  `conf=0.05`, `iou=0.50`, `imgsz=1280`;
- пишет `.txt` рядом с каждым `.jpg`;
- проверяет, что у каждого изображения есть `.txt`;
- скачивает zip `cvat_video_frames_prelabelled_full.zip`.

По умолчанию оставь:

```python
ROTATE = "auto"
AUTO_ORIENT = True
```

`ROTATE="auto"` означает: на этапе нарезки кадры сохраняются как есть, а
ориентация выбирается перед авторазметкой детектором. Если точно знаешь, что
весь набор снят одной камерой, можно принудительно поставить `ccw`, `cw`,
`180` или `none`.

Если надо больше кадров для плотной разметки, поменяй:

```python
EVERY_SECONDS = 1.0
```

Для первого большого прохода оставь `2.0`: это хороший баланс между качеством
и объемом ручной разметки.

## 3. Вернуть zip на ноутбук

В рабочей папке ветки:

```bash
cd /Users/cute/Lenta_Tech/ultralytics-annotation
```

Распакуй скачанный zip так, чтобы получилась папка:

```text
cvat_video_frames/
  <video_scene>/000000.jpg
  <video_scene>/000000.txt
  photos_extra__<group>/<photo>.jpg
  photos_extra__<group>/<photo>.txt
```

Быстрая проверка:

```bash
find cvat_video_frames -name "*.jpg" | wc -l
find cvat_video_frames -name "*.txt" | wc -l
```

Числа должны совпадать. Пустой `.txt` - это нормально: кадр/фото проверен
моделью, ценник не найден, такой negative полезен.

## 4. Залить в CVAT

Поднять CVAT:

```bash
cd third_party/cvat
docker compose up -d
cd ../..
```

Собрать CVAT seed-файлы:

```bash
.venv/Scripts/python.exe projects/price_tag_pipeline/scripts/frames_to_cvat.py \
  --frames-dir cvat_video_frames \
  --out cvat_seeds_full
```

Создать/заполнить задачи в CVAT:

```bash
.venv/Scripts/python.exe projects/price_tag_pipeline/scripts/cvat_bootstrap_photos.py \
  --manifest cvat_seeds_full/manifest.json \
  --user admin \
  --password ***
```

Если задачи уже существуют и надо обновить только предразметку:

```bash
.venv/Scripts/python.exe projects/price_tag_pipeline/scripts/cvat_bootstrap_photos.py \
  --manifest cvat_seeds_full/manifest.json \
  --user admin \
  --password *** \
  --replace
```

Важно: `--replace` заменяет аннотации, но не перезаливает изображения. Если
поменял `ROTATE`, `EVERY_SECONDS` или состав кадров, старую CVAT task лучше
удалить и загрузить заново.

## 5. Как размечать в CVAT

Открой:

```text
http://localhost:8080
```

Дальше:

1. Project `Lenta price tags`.
2. Открыть task.
3. Открыть именно job внутри task.
4. Проверить каждый кадр:
   - удалить ложные боксы;
   - подтянуть неточные боксы;
   - добавить пропущенные ценники;
   - кадры без ценников оставить пустыми.
5. Часто жать `Ctrl+S`.

Размечаем только один класс:

```text
price_tag
```

OCR-поля и атрибуты сейчас не заполняем. Это detector-first набор для
улучшения поиска ценников; текст/QR живут в другом этапе пайплайна.

## 6. Забрать проверенную разметку из CVAT

Список task names можно взять из `cvat_seeds_full/manifest.json`.

Пример:

```bash
.venv/Scripts/python.exe projects/price_tag_pipeline/scripts/cvat_pull_pack.py \
  --tasks new_store_01,new_store_02,photos_extra__glare_cases \
  --images-dir cvat_video_frames \
  --set-name full_cvat_20260518 \
  --user admin \
  --password *** \
  --into data/raw_photos
```

Результат:

```text
data/raw_photos/
  frames/full_cvat_20260518/*.jpg
  annotations/labels/full_cvat_20260518/*.txt
  annotations/classes.txt
```

Параллельно скрипт сделает zip в `cvat_exports/full_cvat_20260518.zip`.

## 7. Собрать train-ready датасет

```bash
.venv/Scripts/python.exe projects/price_tag_pipeline/scripts/prepare_data.py \
  --raw data/raw_photos \
  --processed data/processed_photos
```

Если нужно объединить старые CSV-видео и новые CVAT-фото в одно обучение,
сначала собери оба processed-набора отдельно, потом тренируй на том YAML,
который соответствует нужному эксперименту. Для detector-only улучшения
обычно достаточно `data/processed_photos/dataset.yaml`.

## 8. Дообучить детектор с новых весов

Если новые веса лежат локально:

```text
data/checkpoints/detector/new_pretrained.pt
```

Запуск:

```bash
.venv/Scripts/python.exe projects/price_tag_pipeline/scripts/train_detector_yolo.py \
  --dataset data/processed_photos/dataset.yaml \
  --model data/checkpoints/detector/new_pretrained.pt \
  --imgsz 1280 \
  --batch 8 \
  --epochs 120 \
  --device 0 \
  --name price_tag_full_cvat_20260518
```

Для Colab уменьши `batch` до `4`, если T4 упирается в память.

После обучения:

```bash
mkdir -p data/checkpoints/detector
cp runs/lenta/price_tag_full_cvat_20260518/weights/best.pt \
  data/checkpoints/detector/best.pt
```

## 9. Мини-чеклист качества перед обучением

Перед запуском train проверь:

- кадры из видео вертикальные, ценники читаются не боком;
- у каждого `.jpg` есть `.txt`;
- в CVAT нет систематического смещения боксов после поворота;
- фото с бликами, смазом, стеклом и пустыми полками действительно добавлены;
- пустые кадры/фото не удалены, они нужны как hard negatives;
- task names из `cvat_pull_pack.py --tasks` совпадают с именами в CVAT;
- новые веса использовались в Colab: в CELL 4 должно быть
  `Using custom weights: .../weights/best.pt`.

## 10. Когда использовать старый notebook

`notebooks/lenta_prelabel_colab.ipynb` оставь для короткого сценария:

1. Ты уже нарезал `cvat_video_frames/` на ноутбуке через
   `slice_video_frames.py`.
2. Нужно только прогнать GPU-предразметку в Colab.

Для полной загрузки новых видео + фото + новых весов используй:

```text
notebooks/lenta_full_data_cvat_colab.ipynb
```
