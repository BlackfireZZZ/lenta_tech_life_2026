export interface ColorMetaCopy {
  label: string;
  swatch: string;
  ring: string;
}

export interface FieldGroupCopy {
  title: string;
  fields: { key: string; label: string }[];
}

export const TAG_SCHEMA_COPY = {
  fallbackTagPrefix: "Ценник #",
  colorMeta: {
    white: { label: "Обычная цена", swatch: "#ffffff", ring: "#d6d3d1" },
    yellow: { label: "Цена по карте", swatch: "#f7d774", ring: "#e0b94a" },
    green: { label: "Промо", swatch: "#86d9a8", ring: "#3fae6f" },
    red: { label: "Акция", swatch: "#f0a3a3", ring: "#dc6a6a" },
  } as Record<string, ColorMetaCopy>,
  displayType: {
    "к": "к · коробка",
    "л": "л · лоток",
    "ш": "ш · штука",
  } as Record<string, string>,
  fieldGroups: [
    {
      title: "Товар",
      fields: [
        { key: "product_name", label: "Наименование" },
        { key: "color", label: "Цвет ценника" },
        { key: "id_sku", label: "Артикул (SKU)" },
        { key: "code", label: "Код зоны выкладки" },
        { key: "additional_info", label: "Доп. информация" },
        { key: "special_symbols", label: "Тип выкладки" },
      ],
    },
    {
      title: "Цены",
      fields: [
        { key: "price_default", label: "Цена без карты" },
        { key: "price_card", label: "Цена по карте" },
        { key: "price_discount", label: "Промо-цена" },
        { key: "discount_amount", label: "Размер скидки" },
      ],
    },
    {
      title: "Идентификация",
      fields: [
        { key: "barcode", label: "Штрихкод" },
        { key: "print_datetime", label: "Дата и время печати" },
      ],
    },
    {
      title: "QR-код",
      fields: [
        { key: "qr_code_barcode", label: "Штрихкод из QR" },
        { key: "price1_qr", label: "Цена 1 (QR)" },
        { key: "price2_qr", label: "Цена 2 (QR)" },
        { key: "price3_qr", label: "Цена 3 (QR)" },
        { key: "price4_qr", label: "Цена 4 (QR)" },
        { key: "action_price_qr", label: "Акционная цена (QR)" },
        { key: "action_code_qr", label: "Код акции (QR)" },
        { key: "wholesale_level_1_count", label: "Опт ур.1 · кол-во" },
        { key: "wholesale_level_1_price", label: "Опт ур.1 · цена" },
        { key: "wholesale_level_2_count", label: "Опт ур.2 · кол-во" },
        { key: "wholesale_level_2_price", label: "Опт ур.2 · цена" },
      ],
    },
  ] as FieldGroupCopy[],
  summary: {
    foundTags: "Найдено ценников",
    byPriceType: "По типу цены",
  },
  tagsTable: {
    name: "Ценник",
    noCard: "Без карты",
    byCard: "По карте",
    barcode: "Штрихкод",
    partial: "(частичный)",
    notRecognized: "не распознан",
  },
  tagFields: {
    absent: "нет на ценнике",
    unrecognized: "не распознано",
  },
} as const;
