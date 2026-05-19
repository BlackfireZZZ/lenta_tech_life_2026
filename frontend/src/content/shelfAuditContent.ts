import type { AlertType } from "@/api/shelfAudit";

export const SHELF_AUDIT_COPY = {
  heroKicker: "Lenta Tech Life 2026 · Дополнительная возможность",
  heroTitle: "Аудит полки",
  heroLead:
    "Тот же проезд робота, что даёт зачётный CSV, попутно находит пустые полки и товары без ценника, и собирает карточку на каждый товар. Полностью автоматически, рядом с зачётным CSV — не вместо него.",
  heroBadges: [
    "Полностью автоматически",
    "Рядом с зачётным CSV",
    "Локально · без облака",
  ],
  load: {
    noData: "Пока нет данных аудита для показа.",
    failed: "Не удалось загрузить данные аудита.",
    noViolations: "Нарушений не найдено на этом видео.",
  },
  statLabels: {
    outOfStock: "Пустые полки",
    missingTagCandidates: "Без ценника (кандидаты)",
    productsRecognized: "Распознано товаров",
    relation: "Ценник ↔ товар",
  },
  summaryHint:
    "Группировка фейсингов в товар убирает ложные срабатывания: {tracks} фейсингов → {cards} карточек; «товар без ценника» {rawMissing} → {missing} на уровне товара. «Пустая полка» — точный сигнал; «без ценника» пока кандидаты (зависит от того, попал ли ценник в кадр рядом с товаром).",
  sections: {
    alerts: "Алерты",
    cards: "Карточки товаров",
  },
  alertsTitle: "Как читать алерты",
  alertsText:
    "Левая колонка — это очередь кандидатов на проблему. «Пустая полка» обычно точный сигнал. «Товар без ценника» — кандидат на проверку: часть таких случаев связана с качеством кадра и тем, что ценник оказался вне зоны видимости в моменте.",
  modelNoteTitle: "Почему есть кандидаты",
  modelNoteText:
    "Для экономии ресурсов используется компактный локальный стек: детектор дообучен на полках Ленты, а распознавание текста работает на небольшой VLM-модели и запасных OCR. Это сознательный баланс скорости и точности; спорные случаи помечаются как кандидаты, а не как жёсткое нарушение.",
  purposeTitle: "Зачем это",
  purposeText:
    "Робот, который и так читает ценники, тем же проездом превращается в аудитора полки: подсвечивает потенциальные потери и нарушения выкладки. Цена, название и штрихкод в карточках берутся из основного распознавания ценника.",
  alertTypeLabel: {
    OUT_OF_STOCK: "Пустая полка",
    MISSING_PRICE_TAG: "Товар без ценника",
  } as Record<AlertType, string>,
  alertDescription: {
    OUT_OF_STOCK: "Ценник распознан, но товара над ним нет — упущенные продажи.",
    MISSING_PRICE_TAG:
      "Кандидат: товар без сопоставленного ценника. Требует проверки — на части видео ценник просто не попал в кадр рядом с товаром.",
  } as Record<AlertType, string>,
  metrics: {
    visibleFramesSuffix: "кадров",
    facingsSuffix: "фейсингов",
    trackPrefix: "трек",
  },
  cardTile: {
    facingsPrefix: "×",
    facingsSuffix: "фейсингов",
    noName: "Название — из распознавания ценника",
    noPrice: "цена — с ценника",
    noBarcode: "штрихкод — с ценника",
    rubSuffix: "₽",
  },
} as const;

export function formatSummaryHint(params: {
  tracks: number;
  cards: number;
  rawMissing: number;
  missing: number;
}): string {
  return SHELF_AUDIT_COPY.summaryHint
    .replace("{tracks}", String(params.tracks))
    .replace("{cards}", String(params.cards))
    .replace("{rawMissing}", String(params.rawMissing))
    .replace("{missing}", String(params.missing));
}
