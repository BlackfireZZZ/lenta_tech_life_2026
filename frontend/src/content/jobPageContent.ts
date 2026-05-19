export const JOB_PAGE_COPY = {
  errors: {
    notFound: "Не удалось найти эту обработку — возможно, сервер перезапустился.",
    predictions: "Не удалось загрузить результаты распознавания.",
    failedDefault: "Обработка завершилась с ошибкой.",
    genericTitle: "Не получилось",
    backUpload: "Загрузить другое видео",
  },
  tableTitle: "Все ценники",
  reviewer: {
    title: "Проверка ценников",
    prevTagTitle: "Предыдущий ценник (←)",
    prevTagButton: "Назад",
    nextTagTitle: "Следующий ценник (→)",
    nextTagButton: "Вперёд",
    playingOverlay: "Воспроизведение — рамки скрыты",
    playTitle: "Воспроизвести",
    pauseTitle: "Пауза",
    pointBestFrameSuffix: "— лучший кадр",
    timelineHint:
      "Каждая точка — лучший кадр ценника: именно по нему распознаны данные и взято время в видео. Нажмите, чтобы открыть; тяните дорожку, чтобы перемотать.",
    cropTitle: "Изображение ценника",
    cropFailed: "Не удалось показать изображение ценника.",
  },
  frameSwitcher: {
    manyInFrame: "В этом кадре несколько ценников",
    prevInFrameTitle: "Предыдущий ценник в кадре",
    nextInFrameTitle: "Следующий ценник в кадре",
    indexDivider: "из",
  },
  heroStats: {
    priceDefault: "Цена без карты",
    priceCard: "Цена по карте",
    barcode: "Штрихкод",
    sku: "Артикул (SKU)",
  },
  groups: {
    valuesSuffix: "значений",
    noData: "нет данных",
  },
  fieldState: {
    absent: "нет на ценнике",
    unrecognized: "не распознано",
  },
  header: {
    newVideo: "Новое видео",
    ready: "Готово",
    uniqueTagsSuffix: "уникальных ценников",
    downloadCsv: "Скачать таблицу (CSV)",
  },
  processing: {
    title: "Обрабатываем видео",
    queued: "В очереди…",
    detect: "Ищем ценники в кадрах…",
    finalize: "Распознаём ценники нейросетью…",
    dedup: "Склеиваем повторы одного ценника…",
    done: "Формируем результат…",
    fallbackMid: "Распознаём поля и штрихкоды…",
    fallbackFinal: "Завершаем…",
    hint:
      "Видео обрабатывается покадрово — это занимает время. Экран обновится автоматически.",
  },
} as const;
