# Глобальное исследование датасетов для роботизированного аудита магазинных полок

**Детекция товаров и ценников, OCR, подсчет facings/количества, OOS/gap detection и planogram compliance**

Подготовлено: 17 мая 2026. Проверка выполнена по публичным страницам датасетов, статьям, репозиториям и описаниям доступности. Крупные закрытые/платные наборы не скачивались целиком; отдельно отмечено, где датасет публичный, коммерческий, требует Kaggle/Roboflow аккаунт или не опубликован как download.

> **Главный вывод:** открытого датасета “робот едет вдоль полок + товары + ценники + OCR + точный stock-depth count + planogram + хорошая SKU-разметка” в одном пакете практически нет. Лучший путь - модульный стек датасетов плюс собственная доменная разметка видео с робота. Для MVP оптимальны SKU-110K/Locount для товаров и facings/count, SHARD/SHAPE для SKU/row/planogram, SOVAR + HITL для ценников, Gap datasets/ROSCH для OOS, и небольшой собственный robot-aisle набор для адаптации камеры, размытия, углов и высоты робота.

## 1. Резюме для принятия решения

Задача пользователя объединяет несколько разных компьютерно-зрительных задач: обнаружить фронтальные товарные единицы на полке, связать их с SKU, найти ценник, прочитать цену/OCR, связать ценник с товаром, оценить количество видимых facings, обнаружить пустые места и сравнить раскладку с планограммой. Исследование показывает, что эти задачи обычно покрываются разными датасетами и статьями, а не одним универсальным набором.

**Практический рейтинг:** для обучения первого рабочего прототипа стоит взять SKU-110K как backbone для dense product detection, Locount как специальный датасет для localization+counting, SHARD/SHAPE как эталонную архитектуру “полки -> товары -> SKU -> планограмма”, HITL/SOVAR для ценников, а ROSCH/Gap datasets как источник идей и разметки для OOS. 7-Eleven Scientific Reports ценен как промышленный референс архитектуры, но датасеты из статьи, судя по публикации, не представлены как публичный download.

| Категория | Лучшие кандидаты | Почему |
| --- | --- | --- |
| Детекция товаров / facings | SKU-110K, Locount | SKU-110K имеет >1.7M bbox в packed shelf scenes; Locount специально вводит localization + counting для групп объектов. |
| SKU recognition / matching | SHAPE, RP2K, RPC, GroZi-3.2K | SHAPE имеет ~46K изображений и ~16K SKU; RP2K/RPC полезны для embedding/recognition, но не всегда shelf-specific. |
| Ценники / price tags | HITL Supermarket Shelves, SOVAR Price Tag Detection | HITL размечает Product и Price; SOVAR дает Roboflow price-tag detector dataset, но небольшой и с неоднородными классами. |
| Пустые места / OOS | Gap Detection, ROSCH | Gap datasets дают masks/regions gap/non-gap; ROSCH ближе всего к роботу в магазине, но датасет не явно выложен как обычный public dataset. |
| Planogram compliance | Shelf Management SHARD/SHAPE, 7-Eleven paper | Обе работы описывают полный production-like pipeline для shelf rows, products, recognition и comparison с planogram. |

> **Оценка “AGI/одна большая модель”:** лучше не делать одну модель “всё сразу”. Для реального магазина и робота устойчивее модульная архитектура: детектор полок, детектор товаров, детектор ценников, OCR, связывание объектов, SKU-embedding retrieval, temporal tracking и бизнес-логика планограммы. Так проще контролировать ошибки, переобучать отдельные модули и добавлять новые SKU.

## 2. Методология проверки

Я оценивал каждый датасет по критериям, которые важны именно для робота, двигающегося вдоль магазинных полок:

- **Домен:** настоящие магазинные полки, видео/роботный viewpoint, плотность товаров, occlusion, блики, motion blur.
- **Разметка:** bbox/segmentation товаров, SKU/EAN/class labels, shelf rows, ценники, OCR текст/цена, gap/OOS, количество/facings.
- **Доступность:** публичный download, лицензия, Kaggle/Roboflow/figshare/GitHub, коммерческие ограничения.
- **Использование в работах:** наличие статьи, benchmark, репозитория, цитирования/реализаций/документации.
- **Пригодность для production:** насколько датасет можно реально использовать для обучения модулей MVP и где нужен собственный сбор данных.

В отчете термины: **facing** - видимая фронтальная единица товара на полке; **stock-depth** - реальное количество товара в глубину, которое по одному RGB-кадру обычно не определяется надежно; **OOS** - out-of-stock/пустая или частично пустая полка; **planogram compliance** - соответствие фактической выкладки заданной планограмме.

## 3. Сравнительная матрица датасетов

Оценки 0-5 являются прикладной экспертной оценкой для конкретной задачи “робот едет вдоль полок, видит товары и ценники, считает и сверяет”. Они не являются универсальной оценкой качества датасета.

| Датасет | Товар/ bbox | SKU/ class | Ценник/OCR | Count/OOS | Робот/видео | Доступ | Итог |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SKU-110K | 5 | 1 | 0 | 3 | 2 | 4 | Лучший старт для dense product/facing detection. |
| SHARD/SHAPE | 4 | 5 | 0 | 2 | 2 | 5 | Лучший открытый research stack для shelf rows + SKU recognition + planogram. |
| 7-Eleven Virtual Shelves | 5 | 5 | 0 | 3 | 2 | 1 | Очень сильная production-архитектура, но не как открытый датасет. |
| Locount | 4 | 3 | 0 | 5 | 1 | 3 | Лучший под localization+counting групп/товаров. |
| GroZi-120 / GroZi-3.2K | 3 | 4 | 0 | 2 | 4 | 3 | Старый, но важный для in-vitro -> in-situ recognition и видео/полок. |
| HITL Supermarket Shelves | 3 | 0 | 4 | 1 | 1 | 5 | Маленький, но ценный старт для Product + Price bbox. |
| SOVAR Price Tag | 1 | 0 | 4 | 0 | 1 | 4 | Небольшой Roboflow датасет для ценников, требует QA. |
| Gap Detection | 1 | 0 | 0 | 4 | 1 | 5 | Специализированные masks/regions gap vs non-gap. |
| ROSCH | 2 | 0 | 0 | 5 | 5 | 2 | Ближайшая работа к автономному роботу, но не полноценный public training pack. |
| Unidata Grocery Shelves | 4 | 2 | 0 | 3 | 1 | 2 | Коммерческий, интересен из-за атрибутов facing/flipped/occluded. |
| RPC | 2 | 5 | 0 | 2 | 0 | 4 | Для checkout/SKU recognition, не для полок/ценников. |
| RP2K | 2 | 5 | 0 | 1 | 1 | 3 | Сильный для fine-grained retail classification/retrieval. |
| Freiburg Groceries | 1 | 2 | 0 | 0 | 1 | 5 | Больше classification/robot grocery benchmark, не shelf audit. |
| Grocery Dataset / WebMarket | 3 | 3 | 0 | 2 | 1 | 3 | Исторически полезные shelf sets, но меньше и старее. |

## 4. Подробные профили датасетов

### 4.1 SKU-110K

**Назначение:** Dense object detection на фотографиях магазинных полок: детектировать каждый видимый товар в очень плотных сценах.

**Размер и разметка:** По Dataset Ninja: 11,689 изображений, 1,723,135 bbox, один класс retail item, все изображения размечены; split: train 8,185, test 2,920, val 584. Изображения собраны в разных магазинах и регионах, аннотации проходили визуальную проверку.

**Что хорошо:** Отлично учит модель разделять соседние товары и считать видимые front-facing единицы. Это именно shelf domain, dense scenes, occlusions, разные углы и освещение.

**Что плохо:** Нет ценников, OCR, привязки к конкретному SKU/EAN, planogram и реального stock-depth. Класс один: “retail item”, поэтому для SKU recognition нужен отдельный retrieval/classification модуль.

**Использование в работах:** Официальная CVPR 2019 работа “Precise Detection in Densely Packed Scenes”; есть официальный GitHub, практические доки Ultralytics и сторонние implementations. Это один из самых узнаваемых shelf detection benchmarks.

**Доступность:** Публичный academic/non-commercial dataset через официальный GitHub/зеркала; формат обычно CSV/bbox, легко конвертируется в COCO/YOLO.

**Вывод для проекта:** Брать обязательно как базовую предобучающую выборку для product/facing detector. Не использовать как единственный датасет, потому что он не знает ценники и SKU.

**Источники:** [S1](#s1), [S2](#s2), [S3](#s3).

### 4.2 Shelf Management: SHARD + SHAPE

**Назначение:** End-to-end shelf monitoring: детекция товаров, распознавание SKU, привязка к shelf rows, planogram compliance.

**Размер и разметка:** SHAPE на Figshare содержит ~46K изображений ~16K SKU в 62 категориях; labels на уровне category и EAN, EAN анонимизированы. В статье/репозитории также описывается SHARD для shelf-row detection.

**Архитектура из работы:** RetinaNet для object detection, Deep Hough Transform для shelf-row semantic lines, MobileNetV3 + triplet loss как embedding extractor, FAISS retrieval, deterministic localization по продуктам и рядам.

**Метрики из репозитория/abstract:** mAP 0.752 для detection, F1 97% для shelf rows, top-1 recognition 93% для MobileNetV3+FAISS.

**Что хорошо:** Очень близко к требуемой архитектуре: товар -> ряд -> позиция -> SKU -> планограмма. Открытый код и данные, SHAPE под CC BY 4.0.

**Что плохо:** Нет ценников/OCR и нет роботного видео. EAN анонимизированы: для коммерческого сопоставления с реальным каталогом потребуется собственная связка SKU/EAN.

**Вывод для проекта:** Это главный research reference для общей архитектуры. Использовать SHAPE для SKU embedding/retrieval и идею SHARD для shelf-row module.

**Источники:** [S4](#s4), [S5](#s5), [S6](#s6).

### 4.3 7-Eleven Taiwan / Virtual Shelves / Planogram Compliance

**Назначение:** Промышленная система planogram compliance для convenience stores: shelf detection, product detection, product classification, stitching/virtual shelves, сравнение с digital planogram.

**Размер и разметка:** В статье указаны три больших custom datasets: 15,232 изображения для shelf detection, 99,135 изображений для product detection и 471 product category со средним 210 изображений на категорию.

**Метрики:** YOLOv8 shelf detection: precision 99.23%, recall 98.93%, mAP@50 99.41%; product detection: precision 94.61%, mAP@50 95.73%; FAN-based hybrid classification: accuracy 99.86%; zero-shot top-1 98.39%, top-5 99.48%.

**Что хорошо:** Очень сильный production benchmark: более 7,000 7-Eleven stores, virtual shelf stitching, handling spatial constraints, planogram matching.

**Что плохо:** Датасеты описаны в статье, но не выглядят как общедоступный download. Нет явной разметки ценников/OCR. Это скорее архитектурный ориентир, чем источник training data.

**Вывод для проекта:** Использовать как образец промышленной архитектуры и метрик, но не рассчитывать на него как на открытый датасет для обучения.

**Источники:** [S7](#s7).

### 4.4 Locount: Rethinking Object Detection in Retail Stores

**Назначение:** Новая задача localization + counting: не просто bbox каждого объекта, а локализовать группы объектов и оценивать число instances внутри группы.

**Размер и разметка:** AAAI/arXiv описывает 50,394 изображений, >1.9M object instances, 140 categories. Изображения 1920x1080, собраны в 28 разных магазинах и квартирах, согласно GitHub summary из grocery detection collection.

**Что хорошо:** Лучше других датасетов подходит к твоему запросу “оценивать количество товара”. Он признает проблему severe occlusion и вводит count как часть задачи.

**Что плохо:** Не фокусируется на ценниках/OCR и не является robot video. Категории не обязательно соответствуют SKU уровня EAN/GTIN. Доступность официального GitLab может требовать ручной проверки/регистрации.

**Использование в работах:** AAAI 2021 paper; имеются сторонние repos на MMDetection и цитирования/обсуждения в новых работах по retail product detection/counting.

**Вывод для проекта:** Добавить в training/eval set для модуля count и robustness при occlusions. Особенно полезен как benchmark для группового подсчета, где bbox каждого отдельного товара ненадежен.

**Источники:** [S20](#s20), [S21](#s21), [S22](#s22), [S13](#s13).

### 4.5 GroZi-120 и GroZi-3.2K

**Назначение:** Fine-grained grocery product recognition: сопоставлять идеальные product images/in-vitro с реальными кадрами shelves/in-situ.

**GroZi-120:** 120 grocery products; для каждого продукта есть in-vitro изображения из web и in-situ кадры из camcorder video внутри grocery store. Это один из первых публичных наборов для такого домена.

**GroZi-3.2K:** Более новая версия/связанная линия: recent preprint summary описывает 3,235 Swiss retail shelf images + reference product images, детальные product detection annotations на shelf images.

**Что хорошо:** Близко к роботу/видео и к задаче recognition in real store. Хорошо для retrieval/embedding: “есть эталон товара, найди его на полке”.

**Что плохо:** Старые данные, ограниченное число продуктов, нет ценников/OCR, качество/разрешение может быть хуже современных камер. Не решает planogram и stock-depth.

**Вывод для проекта:** Использовать как вспомогательный benchmark для product recognition/retrieval и temporal/video-like evaluation, но не как основной набор для современной полочной аналитики.

**Источники:** [S12](#s12), [S13](#s13), [S25](#s25).

### 4.6 Grocery Dataset / Toward Retail Product Recognition on Grocery Shelves

**Назначение:** Shelf images + product/logo images для recognition на магазинных полках.

**Размер и разметка:** GitHub указывает 354 grocery shelf images, собранные с ~40 groceries и 4 camera types; 10 product categories; product images и cropped brand/logo images; research-only license.

**Что хорошо:** Имеет планограммные/категорийные поля в naming convention, реальные камеры, исторически важен для shelf recognition.

**Что плохо:** Маленький по современным меркам, research-only, нет ценников/OCR и не подходит как главный training set для detector.

**Вывод для проекта:** Полезен для прототипных экспериментов и historical comparison, но для production его нужно дополнять SKU-110K/SHAPE/своими кадрами.

**Источники:** [S17](#s17).

### 4.7 Gap Detection Datasets

**Назначение:** Детекция пустых мест на полках: gap/non-gap regions/masks на существующих shelf datasets.

**Размер и разметка:** GitHub описывает annotations для Grocery Products, WebMarket и GroZi-120: 305, 98 и 50 shelf images соответственно; примерно 60% train / 40% test. Связанная статья: Pattern Recognition 2022.

**Что хорошо:** Прямо решает OOS/gap часть задачи, которую обычные product detectors не покрывают. Можно учить отдельный segmentation/detection module для пустых зон.

**Что плохо:** Датасет небольшой, старый, не SKU-level, нет price tags. Gap не всегда означает out-of-stock конкретного SKU без планограммы и ценника.

**Вывод для проекта:** Использовать как seed для OOS/gap detector, но обязательно доразметить свои robot frames с реальными пустыми слотами и планограммой.

**Источники:** [S14](#s14).

### 4.8 SOVAR Price Tag Detection / Roboflow

**Назначение:** Price tag / label detector для retail shelf images.

**Размер и разметка:** Roboflow project page указывает 477 open-source Pricetags images; dataset version page показывает 1073 total images после augmentations, split train 83%, valid 12%, test 4%, resize 640x640 и набор augmentations.

**Классы:** Project page перечисляет классы: labels_prices_products, labels-prices-products, pricetag и т.п.; это указывает на неоднородность labels и потенциальную необходимость нормализации схемы.

**Что хорошо:** Редкий открытый источник именно для price tag detection; экспорт в YOLO/COCO через Roboflow удобен.

**Что плохо:** Маленький, Roboflow community dataset без полноценной academic validation. Не содержит надежный OCR ground truth по цене и товару; классы нужно почистить.

**Вывод для проекта:** Можно использовать как initial detector for price tag crops, но перед production обязательно провести manual QA labels и добавить свои ценники из целевых магазинов.

**Источники:** [S8](#s8), [S9](#s9).

### 4.9 Humans in the Loop Supermarket Shelves Dataset

**Назначение:** Открытый sample dataset для product и price tag detection.

**Размер и разметка:** HITL описывает 45 copyright-free images of shelves; отдельная страница datasets указывает 11,743 bounding boxes с классами Product и Price.

**Что хорошо:** Очень ценен именно потому, что вместе размечает Product и Price. Лицензия позволяет academic and commercial usage. Хорош для проверки label schema и tiny benchmark.

**Что плохо:** Слишком маленький для обучения серьезной модели. Нет SKU/EAN, OCR текста, привязки ценника к товару и robot video.

**Вывод для проекта:** Использовать как эталон схемы “Product + Price bbox” и sanity check для detector; не ждать хорошей generalization без большого собственного набора.

**Источники:** [S10](#s10), [S11](#s11).

### 4.10 Unidata Grocery Shelves Dataset

**Назначение:** Коммерческий набор real-world grocery shelf images для object detection/classification и shelf monitoring.

**Размер и разметка:** Unidata заявляет 5,000+ high-resolution shelf images, XML labels, product attributes: facing, flipped, occluded; data collected via crowdsourcing from real stores.

**Что хорошо:** Атрибуты facing/flipped/occluded практически полезны для shelf audit. Коммерческий поставщик может предоставить стабильную документацию и договорные гарантии.

**Что плохо:** Полная версия платная/по запросу; неизвестна детализация SKU/EAN и наличие price tags/OCR. До покупки нужно запросить sample XML, label taxonomy и лицензию на model training/commercial deployment.

**Вывод для проекта:** Рассматривать как коммерческое ускорение, если бюджет позволяет. Запрашивать sample и проводить label audit перед покупкой.

**Источники:** [S19](#s19).

### 4.11 RPC: Retail Product Checkout Dataset

**Назначение:** Automatic checkout: распознать товары на checkout images, а не аудит полок.

**Размер и разметка:** Project page: train exemplar images 53,739; validation checkout images 6,000 с 73,602 objects; test checkout images 24,000 с 294,333 objects. License CC BY-NC-SA 4.0; dataset on Kaggle/Baidu.

**Что хорошо:** Большой fine-grained retail dataset, хорош для SKU/product recognition, embeddings, checkout-style clutter, evaluation tooling and leaderboard.

**Что плохо:** Не shelf aisle, не robot video, не ценники. Объекты обычно лежат на кассе/столе, а не стоят рядами на полке.

**Вывод для проекта:** Использовать для product recognition pretraining/embedding, но не для обучать product detector на полках как основной domain.

**Источники:** [S15](#s15), [S16](#s16).

### 4.12 RP2K

**Назначение:** Fine-grained retail product classification/retrieval на изображениях товаров в магазинах.

**Размер и разметка:** arXiv summary: 500K+ images, 2,000 retail product classes, captured manually in physical retail stores, with annotations such as sizes, shapes and flavors/scents.

**Что хорошо:** Сильный источник для SKU-level retrieval/classification на большом числе продуктов; более “shelf-like” чем checkout-only data.

**Что плохо:** Основная задача - classification, не detection; нет ценников/OCR/полочных рядов/планограммы. Нужно проверять доступность данных и лицензию на текущем сайте.

**Вывод для проекта:** Использовать как дополнительный pretraining set для product embeddings, если доступен. Не заменяет shelf detection datasets.

**Источники:** [S28](#s28).

### 4.13 Freiburg Groceries Dataset

**Назначение:** Классификация grocery categories для robotics/domestic environments.

**Размер и разметка:** arXiv: 5,000 images, 25 classes, collected in real-world settings at stores and apartments; at least 97 images per class; baseline CaffeNet mean accuracy 78.9%.

**Что хорошо:** Хороший простой benchmark для grocery category recognition и робототехнических экспериментов.

**Что плохо:** Не shelf audit: нет плотных bbox товаров, нет ценников, нет SKU/EAN, нет counting и planogram.

**Вывод для проекта:** Подходит только как слабое pretraining/classification baseline. Для текущей задачи не ключевой.

**Источники:** [S18](#s18).

### 4.14 WebMarket и Generic SKU Detection Benchmark

**Назначение:** Ранний supermarket retrieval/detection benchmark; later community annotations для generic SKU detection.

**Размер и разметка:** WebMarket paper: 3,153 images in a Coles supermarket. ParallelDots benchmark добавляет/нормализует generic product annotations для WebMarket, GP, CAPG-GP, Holoselecta, TobaccoShelves.

**Что хорошо:** Полезно для low-data dense detection benchmarks и проверки способности переноситься между старыми shelf datasets.

**Что плохо:** Данные старые, ограниченный домен, нет ценников/OCR и полноценного SKU/planogram for production.

**Вывод для проекта:** Использовать как дополнительный evaluation set, не как главный training source.

**Источники:** [S26](#s26), [S27](#s27).

### 4.15 ROSCH: Autonomous Mobile Robot for OOS Detection

**Назначение:** Роботизированное обнаружение пустых/частично пустых полок в супермаркете.

**Размер и разметка:** ICCV Workshop page: deep-learning detector validated on about 2,000 manually annotated images, 900 acquired by the authors in three supermarkets in Italy; system tested in a supermarket in Salerno during working time.

**Что хорошо:** Ближайшая работа к твоему сценарию: автономный mobile robot, ROS, реальные супермаркеты, OOS detection, speed comparison with human operator.

**Что плохо:** Похоже, датасет не упакован как публичный download; он больше подтверждает архитектуру/подход, чем дает готовые training данные. Нет ценников/OCR/SKU count.

**Вывод для проекта:** Использовать как архитектурный и экспериментальный ориентир для robot collection setup и OOS module. Для данных - писать авторам или собирать собственное видео.

**Источники:** [S23](#s23).

## 5. Что говорят работы и обзоры: качество и ограничения

Современный обзор Melek et al. 2024 прямо формулирует проблему: product recognition on grocery shelf images полезен для inventory, stocking status, price accuracy and customer experience, но остается сложным из-за трудности получения и обновления датасетов и огромного масштаба товаров. Это подтверждает, что “идеальный датасет” быстро устаревает: новые упаковки, акции, сезонные товары и локальные SKU требуют постоянной доменной адаптации.

В работах, наиболее близких к production, применяется не одна модель, а pipeline. Shelf Management использует object detection + shelf-row detection + embedding retrieval + deterministic localization. 7-Eleven строит virtual shelves, использует multi-image stitching и несколько моделей detection/classification. ROSCH добавляет мобильную робототехнику и проверяет OOS на реальном магазине. Это показывает общий industry pattern: modularity beats monolith.

### Главные слабые места всех публичных наборов

- **Ценники и OCR:** почти нет больших академических datasets, где у ценника есть bbox, OCR text, price value и привязка к SKU.
- **Связь price tag -> product:** даже при наличии bbox Product и Price нужна отдельная spatial/semantic association разметка: какой ценник относится к какому товару.
- **Stock-depth:** количество товара в глубину не видно из одного RGB image. Нужны RGB-D, несколько углов, temporal tracking или априорные правила по типу упаковки/глубине полки.
- **Домен робота:** datasets со статичных фото не покрывают blur, rolling shutter, высоту камеры, наклоны, reflections и скорость движения робота.
- **SKU drift:** упаковки и SKU постоянно меняются; static academic dataset быстро устаревает.

## 6. Рекомендованная архитектура для MVP

Рекомендуемая система должна работать как набор контролируемых модулей. Ниже - архитектура, которую можно обучать по найденным датасетам и затем адаптировать на собственных кадрах робота.

| Модуль | Датасеты для старта | Выход | Комментарий |
| --- | --- | --- | --- |
| Frame selection / stitching | 7-Eleven paper как reference, свои robot videos | стабильные keyframes, панорама полки | Нужно уменьшить motion blur и покрыть длинные полки несколькими кадрами. |
| Shelf row detection | SHARD / Shelf Management, 7-Eleven shelf dataset idea | bbox/lines shelf rows | Нужен для привязки товара к ряду и планограмме. |
| Product/facing detector | SKU-110K, Locount, Unidata sample | bbox видимых товаров | Считать visible facings; разделять плотные одинаковые товары. |
| SKU recognizer / retrieval | SHAPE, RP2K, RPC, GroZi-3.2K | sku_id / nearest catalog item | Лучше embedding retrieval + FAISS/Qdrant, а не fixed classifier. |
| Price tag detector | HITL, SOVAR, свои ценники | bbox ценников | Ценники отличаются по магазинам, стране, шрифтам, promo layouts. |
| OCR price parser | свои crops + synthetic labels | price, currency, promo text, unit price | Нужны PaddleOCR/TrOCR/Donut + rule-based postprocessing. |
| Tag-product association | собственная разметка | linked_sku_id per price tag | Критически важно: геометрия + ряд + ближайший товар + planogram prior. |
| Gap/OOS detector | Gap Detection, ROSCH, свои OOS frames | empty slots, partial gaps | Gap без planogram не говорит, какого товара не хватает. |
| Counting | SKU-110K + Locount + temporal tracking | visible count/facings, low-stock estimate | Real stock-depth надежно только с RGB-D/multiview/правилами. |
| Planogram compliance | Shelf Management, 7-Eleven paper | wrong/missing/misplaced products | Сравнить фактическую виртуальную полку с expected planogram. |

## 7. Схема собственной разметки, которая нужна поверх публичных датасетов

Чтобы система реально работала на роботе, нужно собрать собственный небольшой, но качественный датасет целевого магазина/страны/формата ценников. На старте достаточно 2,000-5,000 keyframes из 5-10 часов роботного видео, но разметка должна быть богаче, чем в стандартных object detection наборах.

| Объект разметки | Поля | Зачем нужно |
| --- | --- | --- |
| shelf_row | bbox или line/polygon, row_id, shelf_section_id | Навигация по полке, association product-price, planogram. |
| product_facing | bbox/polygon, sku_id, row_id, facing_id, occluded/truncated, confidence QA | Детекция товара, visible facings, recognition, misplacement. |
| price_tag | bbox/polygon, raw_ocr_text, price, currency, promo_type, unit_price, linked_sku_id | Детекция ценника, OCR, проверка цены и связка с товаром. |
| gap / empty_slot | bbox/polygon, row_id, expected_sku_id if known, severity | OOS/low-stock, пустые места, планограмма. |
| frame/camera | timestamp, robot_pose, camera intrinsics, speed, blur flag, lighting flag | Temporal tracking, stitching, debug ошибок робота. |
| planogram reference | expected row, expected sku order, expected facings, price source | Сравнение фактической полки с планом. |

### Минимальный JSON-like формат

```json
{
  "image_id": "frame_000123",
  "robot_pose": {"x": 12.3, "y": 4.1, "theta": 1.57},
  "shelf_rows": [{"bbox": [x1,y1,x2,y2], "row_id": 2}],
  "products": [{"bbox": [x1,y1,x2,y2], "sku_id": "EAN_or_internal", "row_id": 2, "occluded": false}],
  "price_tags": [{"bbox": [x1,y1,x2,y2], "ocr_text": "2.49", "price": 2.49, "linked_sku_id": "EAN_or_internal"}],
  "gaps": [{"bbox": [x1,y1,x2,y2], "expected_sku_id": "EAN_or_internal", "severity": "partial"}]
}
```

## 8. Рекомендуемый план экспериментов

| Фаза | Что сделать | Критерии успеха |
| --- | --- | --- |
| 0. Audit datasets | Скачать/подключить SKU-110K, SHAPE, HITL, SOVAR, Gap; проверить license; конвертировать в COCO/YOLO. | Единый registry datasets, reproducible scripts, 20-50 sample images inspected manually. |
| 1. Baseline detection | YOLOv8/YOLOv10/RT-DETR on SKU-110K + fine-tune на 500-1000 собственных frames. | mAP@50/95 по product_facing; визуально нет склеивания соседних товаров. |
| 2. Price tags | Train price-tag detector на HITL+SOVAR + свои ценники; добавить OCR parser. | Recall ценников >95% на целевых shelves; price parse accuracy отдельно. |
| 3. SKU embedding | Train/finetune embedding on SHAPE/RP2K/RPC + собственный catalog packshots. | Top-1/Top-5 retrieval по целевому SKU catalog; graceful unknown detection. |
| 4. Association | Разметить linked_sku_id для ценников; учить/править rules: same row, nearest x-range, planogram prior. | Ошибка связки price->SKU ниже бизнес-порога. |
| 5. OOS/count | Gap detector + Locount-like count head + temporal tracking. | Visible facings MAE; OOS precision/recall; low-stock alerts не создают лишний шум. |
| 6. Robot validation | Прогон робота в проходе: разные скорости/освещение/люди/блики. | End-to-end report per shelf section: SKU, price, facings, OOS, wrong price. |

## 9. Риски и вопросы перед стартом

- **Лицензии:** SKU-110K часто ограничен academic/non-commercial; RPC - CC BY-NC-SA; Grocery Dataset - research only; Unidata - paid license. Для коммерческого продукта нужно юридически проверить каждый источник.
- **Качество Roboflow community datasets:** SOVAR полезен, но labels/classes могут быть неоднородны. Не обучать без ручного QA.
- **Цена и OCR:** OCR ground truth придется разметить самостоятельно; public price-tag datasets почти не дают связку “price value -> SKU”.
- **Глубина полки:** если нужен реальный stock count, планируй RGB-D/мультикамеру/мультиугловой проезд; иначе честно называй метрику “visible facings/estimated stock”.
- **SKU catalog:** production quality зависит от актуального каталога с packshot, EAN/GTIN, размерами упаковки и допустимыми ценниками.
- **Planogram source:** без цифровой планограммы gap detector не знает, какой SKU должен быть на пустом месте.

## 10. Финальная рекомендация

Для твоей задачи я бы не стал искать “один идеальный датасет”. Его, скорее всего, не существует в открытом доступе. Лучшее решение - собрать обучающий стек: **SKU-110K** для плотной детекции товаров, **Locount** для подсчета/occlusion, **SHARD/SHAPE** для shelf-row/SKU/planogram логики, **HITL + SOVAR** для ценников, **Gap Detection + ROSCH** для OOS/robot approach, затем собрать собственный маленький датасет целевых проходов с робота.

> С точки зрения MVP порядок работ такой: сначала получить стабильный product/facing detector и price-tag detector, потом добавить OCR и association, затем SKU retrieval, а уже после - OOS/counting и planogram compliance. Это снизит риски: каждый модуль можно измерять и улучшать отдельно.

## 11. Источники и проверенные страницы

Ссылки даны как проверенные публичные страницы/статьи/репозитории. Дата доступа: 17 мая 2026.

<a id="s1"></a>
- **[S1] SKU-110K GitHub repository**  
  официальный код и датасет CVPR 2019; описывает Soft-IoU, EM-Merger и SKU-110K как benchmark для packed retail shelf images  
  <https://github.com/eg4000/sku110k_cvpr19>
<a id="s2"></a>
- **[S2] SKU110K dataset summary - Dataset Ninja**  
  11,689 изображений, 1,723,135 размеченных объектов, один класс retail item, split train/test/val; описание ручной проверки аннотаций  
  <https://datasetninja.com/sku110k>
<a id="s3"></a>
- **[S3] Ultralytics SKU-110K docs**  
  практическая документация для обучения YOLO на SKU-110K; подтверждает популярность benchmark для detection  
  <https://docs.ultralytics.com/datasets/detect/sku-110k>
<a id="s4"></a>
- **[S4] Shelf Management GitHub repository**  
  код и аннотации SHARD/SHAPE; статья указывает RetinaNet 0.752 mAP, Deep Hough F1 97%, MobileNetV3+FAISS top-1 93%  
  <https://github.com/rokopi-byte/shelf_management>
<a id="s5"></a>
- **[S5] SHAPE - Figshare dataset**  
  ~46K изображений, ~16K SKU, 62 категории, fine-grained labels, anonymized EAN, CC BY 4.0  
  <https://figshare.com/articles/dataset/SHAPE_-_SHelf_mAnagement_Product_datasEt/24100704>
<a id="s6"></a>
- **[S6] Shelf Management paper - Expert Systems with Applications**  
  data/code availability and paper metadata for the Shelf Management system  
  <https://www.sciencedirect.com/science/article/pii/S0957417424015021>
<a id="s7"></a>
- **[S7] 7-Eleven Scientific Reports paper**  
  15,232 shelf detection images, 99,135 product detection images, 471 categories; YOLOv8 shelf/product detection and virtual shelves  
  <https://www.nature.com/articles/s41598-025-27773-5>
<a id="s8"></a>
- **[S8] SOVAR price tag detection - Roboflow dataset v1**  
  Roboflow page with 1073 total images in dataset version, train/valid/test split and augmentations  
  <https://universe.roboflow.com/sovar/price-tag-detection-r5jlv/dataset/1>
<a id="s9"></a>
- **[S9] SOVAR price tag detection - Roboflow project**  
  project page lists 477 open-source Pricetags images, classes and pre-trained model/API  
  <https://universe.roboflow.com/sovar/price-tag-detection-r5jlv>
<a id="s10"></a>
- **[S10] Humans in the Loop Supermarket Shelves dataset**  
  open access sample for product and price tag detection, 45 copyright-free images  
  <https://humansintheloop.org/resources/datasets/supermarket-shelves-dataset/>
<a id="s11"></a>
- **[S11] Humans in the Loop datasets list**  
  45 images, 11,743 bounding boxes, classes Product and Price  
  <https://humansintheloop.org/resources/datasets/>
<a id="s12"></a>
- **[S12] GroZi-120 paper PDF**  
  120 grocery products; in-vitro web images and in-situ camcorder video collected inside a grocery store  
  <https://www.michelemerler.com/papers/grozi_cvprw07.pdf>
<a id="s13"></a>
- **[S13] Object Detection Datasets in the Grocery Product Domain - GitHub**  
  collection of grocery object detection datasets; summarizes GroZi120, GroZi-3.2K, Freiburg, SKU110K, Holoselecta, Locount  
  <https://github.com/tobiagru/ObjectDetectionGroceryProducts>
<a id="s14"></a>
- **[S14] Gap detection datasets GitHub**  
  gap/non-gap annotations for Grocery Products, WebMarket and GroZi-120; train/test splits and Pattern Recognition paper reference  
  <https://github.com/gapDetection/gapDetectionDatasets>
<a id="s15"></a>
- **[S15] RPC project page**  
  RPC dataset overview, license, image/object/category counts, Kaggle link, evaluation tool and leaderboard  
  <https://rpc-dataset.github.io/>
<a id="s16"></a>
- **[S16] RPC arXiv page**  
  paper abstract describing checkout images, exemplar images, annotations and benchmark role  
  <https://arxiv.org/abs/1901.07249>
<a id="s17"></a>
- **[S17] Grocery Dataset GitHub**  
  354 shelf images, product/logo images, 10 categories, four camera types; research-only license  
  <https://github.com/gulvarol/grocerydataset>
<a id="s18"></a>
- **[S18] Freiburg Groceries arXiv page**  
  5,000 images, 25 classes, real-world stores/apartments; baseline CaffeNet 78.9%  
  <https://arxiv.org/abs/1611.05799>
<a id="s19"></a>
- **[S19] Unidata Grocery Shelves dataset page**  
  commercial dataset: 5,000+ shelf images, XML labels, facing/flipped/occluded attributes, real stores, sample/full license model  
  <https://unidata.pro/datasets/grocery-shelves/>
<a id="s20"></a>
- **[S20] Locount AAAI/OpenReview page**  
  50,394 images, >1.9M object instances, 140 categories; localization+counting task  
  <https://ojs.aaai.org/index.php/AAAI/article/view/16178>
<a id="s21"></a>
- **[S21] Locount arXiv page**  
  Locount task: localize groups of objects and estimate number of instances; dataset link and benchmark description  
  <https://arxiv.org/abs/2003.08230>
<a id="s22"></a>
- **[S22] Locount GitLab page**  
  official project tree for AAAI 2021 Locount dataset  
  <https://isrc.iscas.ac.cn/gitlab/research/locount-dataset/-/tree/master>
<a id="s23"></a>
- **[S23] ROSCH ICCV Workshop open-access paper**  
  autonomous mobile robot for OOS detection; ~2000 manually annotated images, 900 from three Italian supermarkets, tested in Salerno  
  <https://openaccess.thecvf.com/content/ICCV2023W/ACVR/html/De_Simone_Autonomous_Mobile_Robot_for_Automatic_out_of_Stock_Detection_in_ICCVW_2023_paper.html>
<a id="s24"></a>
- **[S24] Melek et al. 2024 exhaustive review**  
  review of datasets/methods for product recognition on grocery shelf images; highlights dataset acquisition/update challenges  
  <https://www.sciencedirect.com/science/article/abs/pii/S0952197624006109>
<a id="s25"></a>
- **[S25] GroZi-3.2K details in recent dataset survey/preprint**  
  describes GroZi-120 and GroZi-3.2K; 3,235 Swiss shelf images and detailed product detection annotations  
  <https://arxiv.org/html/2411.10591v1>
<a id="s26"></a>
- **[S26] WebMarket / Where is the Weet-Bix PDF**  
  WebMarket with 3,153 supermarket images collected in Coles in 2007  
  <https://users.cecs.anu.edu.au/~wanglei/My_papers/yuhang_accv07.pdf>
<a id="s27"></a>
- **[S27] ParallelDots generic SKU detection benchmark**  
  additional generic product annotations for WebMarket, GP and other datasets, converted into common evaluation format  
  <https://github.com/ParallelDots/generic-sku-detection-benchmark>
<a id="s28"></a>
- **[S28] RP2K arXiv page**  
  500K+ shelf product images, 2,000 product classes; fine-grained classification/retrieval, not price-tag detection  
  <https://arxiv.org/abs/2006.12634>

---

## 12. Расширение списка — найдено веб-поиском (добавлено 2026-05-17, агент)

Дополнения к исследованию выше. Эти источники **отсутствовали** в исходном
документе и закрывают самое слабое место для *нашей* задачи — Russian OCR и
**barcode** (а barcode — это P0: главный ключ матчинга строки с GT, см.
[`../hackathon/briefing.md`](../hackathon/briefing.md) §5). Оценки в той же
шкале 0-5 и для той же задачи, что в §3.

| Датасет | Товар | SKU | Ценник/OCR | Count/OOS | Робот/видео | Доступ | Зачем нам |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RusTitW | 0 | 0 | 4 | 0 | 1 | 4 | Единственный человеко-размеченный набор русского текста *in-the-wild* — прямой домен OCR ценников. |
| BarBeR | 0 | 0 | 3 | 0 | 1 | 5 | 8 748 размеченных barcode-изображений + бенчмарк локализации. Barcode = P0-ключ матчинга. |
| ABBYY barcode benchmark | 0 | 0 | 3 | 0 | 1 | 5 | Synthetic+real barcode detection; дополняет BarBeR, удобен для аугментаций под motion blur. |
| OCR-Cyrillic-Printed-8/-6 | 0 | 0 | 4 | 0 | 0 | 5 | До 1M синтетических изображений печатного кириллического текста — претрейн recognizer'а на RU-цифрах/тексте. |
| KORIE | 0 | 0 | 4 | 0 | 0 | 4 | Не RU и чеки ≠ ценники, но эталон **IE-постановки** (detection→OCR→information extraction под печатными артефактами). Архитектурный референс, как 7-Eleven. |
| Russian OCR Image Corpus | 0 | 0 | 3 | 0 | 0 | 2 | 1000 печатных RU-изображений 11 бытовых сцен. Коммерческий, по запросу — как Unidata. |

### 12.1 Профили

**RusTitW — Russian Language Text in-the-Wild.** Крупный человеко-размеченный
набор для распознавания русского текста в естественных сценах; создан именно
потому, что для русского (в отличие от английского) хороших наборов нет.
Синтетический генератор и данные открыты. Для нас — лучший внешний сигнал под
Cyrillic-OCR, дополняет синтетику Tier 0 из [`datasets.md`](./datasets.md).
Источники: [S29](#s29), [S30](#s30).

**BarBeR — Barcode Benchmark Repository (Pattern Recognition 2024).**
Открытый бенчмарк локализации 1D/2D штрихкодов: ~8 748 размеченных
изображений, несколько алгоритмов, стандартные метрики. **Критично:** в этом
хакатоне barcode — приоритет №1 матчинга строки с GT (см. briefing §5), а
«12 цифр из 13 лучше, чем пусто». Уже упомянут как Tier 3 в
[`datasets.md`](./datasets.md) — здесь канонический репозиторий и метрики.
Источники: [S31](#s31), [S32](#s32).

**ABBYY barcode_detection_benchmark.** Бенчмарк детекции штрихкодов на
synthetic + real данных (Springer 2020). Полезен как второй источник под
barcode-модуль и для генерации размытых/повёрнутых штрихкодов под
robot-motion. Источник: [S33](#s33).

**DonkeySmall/OCR-Cyrillic-Printed-8 и -6 (HuggingFace).** Синтетика печатного
кириллического текста, до ~1M изображений. Ценники печатные → этот набор
хорош для предобучения/дообучения recognizer'а на русских цифрах/строках до
доменного дообучения на наших кропах. Источники: [S34](#s34), [S35](#s35).

**KORIE (MDPI Mathematics, янв 2026).** 748 корейских торговых чеков,
multi-task: scene-text detection + OCR + information extraction под
термопечатными артефактами (выцветание, полосы, заломы). Не наш язык и не
ценники, но это свежий эталон **постановки IE-метрики и пайплайна** «найди →
прочитай → извлеки структуру», аналогичный по роли 7-Eleven в §4.3.
Источник: [S36](#s36).

**Russian OCR Image Corpus (DataoceanAI).** ~1000 печатных русских
изображений, 11 бытовых категорий, собрано в России. Коммерческий, по
запросу — рассматривать как платное ускорение, требовать sample и лицензию,
по аналогии с Unidata (§4.10). Источник: [S37](#s37).

### 12.2 Что это меняет для нашего стека

- **Barcode-модуль повышается до P0-датасета.** Добавить BarBeR (+ABBYY) в
  registry: обучить/проверить детектор+декодер штрихкода до OCR-текста.
  Это напрямую тянет финальную метрику через ключ матчинга.
- **Russian-OCR претрейн.** Цепочка: OCR-Cyrillic-Printed (синтетика) →
  RusTitW (in-the-wild RU) → наши кропы ценников (доменное дообучение).
- **IE-постановка.** KORIE/7-Eleven — как мерить и собирать пайплайн
  «detection→OCR→structured fields», а не отдельные буквы.
- Counting/facings/OOS уже широко покрыт §4.4/§4.7/§6/§8 — это датасетная
  база для **киллер-фичи** (см. [`../strategy.md`](../strategy.md) §12).

### 12.3 Источники (проверено 2026-05-17)

<a id="s29"></a>
- **[S29] RusTitW arXiv 2303.16531** — Russian Language Text Dataset for Visual Text in-the-Wild Recognition  
  <https://arxiv.org/abs/2303.16531>
<a id="s30"></a>
- **[S30] RusTitW — Kaggle dataset**  
  <https://www.kaggle.com/datasets/hardtype/rustitw-russian-language-visual-text-recognition>
<a id="s31"></a>
- **[S31] BarBeR — GitHub (Henvezz95/BarBeR)** — ~8 748 размеченных barcode-изображений, алгоритмы + метрики  
  <https://github.com/Henvezz95/BarBeR>
<a id="s32"></a>
- **[S32] BarBeR — Pattern Recognition 2024 (ACM/Springer)**  
  <https://dl.acm.org/doi/10.1007/978-3-031-78447-7_13>
<a id="s33"></a>
- **[S33] ABBYY barcode_detection_benchmark — GitHub** — synthetic + real barcode detection (Springer 2020)  
  <https://github.com/abbyy/barcode_detection_benchmark>
<a id="s34"></a>
- **[S34] DonkeySmall/OCR-Cyrillic-Printed-8 — HuggingFace** — ~1M синтетических изображений печатной кириллицы  
  <https://huggingface.co/datasets/DonkeySmall/OCR-Cyrillic-Printed-8>
<a id="s35"></a>
- **[S35] DonkeySmall/OCR-Cyrillic-Printed-6 — HuggingFace**  
  <https://huggingface.co/datasets/DonkeySmall/OCR-Cyrillic-Printed-6>
<a id="s36"></a>
- **[S36] KORIE — MDPI Mathematics 14(1):187 (январь 2026)** — Korean retail receipts detection/OCR/IE benchmark  
  <https://www.mdpi.com/2227-7390/14/1/187>
<a id="s37"></a>
- **[S37] Russian OCR Image Corpus — DataoceanAI** — ~1000 печатных RU-изображений, 11 категорий (commercial)  
  <https://dataoceanai.com/datasets/ocr/russian-ocr-image-dataset/>
