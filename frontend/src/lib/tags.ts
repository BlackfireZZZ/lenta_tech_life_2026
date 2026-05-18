// Per-tag presentation logic shared by the review screen. Mirrors the
// graded contract in backend/app/api/v1/schemas/job.py (CSV_COLUMNS /
// SUBSTANTIVE_FIELDS) and the three task.md §3.3 field states.
//
// (Tracked source — see the note in ./utils.ts about the .gitignore fix.)
import { ABSENT, type TagPrediction } from "@/api/jobs";

/** task.md §5.1 — a tag "passes" at ≥80% substantive fields decided. */
export const PASS_THRESHOLD = 0.8;

/**
 * The three task.md §3.3 states, kept visually distinct (confusing the last
 * two loses points):
 *  - `value`         — a recognized value
 *  - `absent`        — "нет": the field is not present on this tag
 *  - `unrecognized`  — "" / missing: present but not read
 */
export type FieldState = "value" | "absent" | "unrecognized";

export function fieldState(value: string | null | undefined): FieldState {
  if (value == null) return "unrecognized";
  const t = value.trim();
  if (t === "") return "unrecognized";
  if (t === ABSENT) return "absent"; // "нет"
  return "value";
}

/**
 * Per-tag completeness = share of *substantive* fields that have been
 * decided (a value OR an explicit "нет" — both are real answers; only an
 * unread field counts against it). This is the honest review-screen proxy
 * for the metric (task.md §5.2); the true score needs the GT.
 */
export function completeness(
  tag: TagPrediction,
  substantive: string[],
): number {
  if (!substantive.length) return 0;
  const decided = substantive.reduce(
    (n, key) => n + (fieldState(tag.fields[key]) !== "unrecognized" ? 1 : 0),
    0,
  );
  return decided / substantive.length;
}

/** Human label for the table/cards — the product name, or a stable fallback. */
export function tagLabel(tag: TagPrediction): string {
  return fieldState(tag.fields.product_name) === "value"
    ? tag.fields.product_name
    : `Ценник #${tag.index + 1}`;
}

export interface ColorMeta {
  label: string;
  swatch: string; // fill
  ring: string; // 1px outline so a near-white chip stays visible
}

// Lenta tag colours encode the price mechanic (docs/hackathon/price-tag-guide).
// Kept light/tinted — data, not decoration (DESIGN: no extra saturated hues).
const COLOR_META: Record<string, ColorMeta> = {
  white: { label: "Белый · обычный", swatch: "#ffffff", ring: "#d6d3d1" },
  yellow: { label: "Жёлтый · по карте", swatch: "#f7d774", ring: "#e0b94a" },
  green: { label: "Зелёный · промо", swatch: "#86d9a8", ring: "#3fae6f" },
  red: { label: "Красный · акция", swatch: "#f0a3a3", ring: "#dc6a6a" },
};

export function colorMeta(color: string): ColorMeta {
  return (
    COLOR_META[(color || "").toLowerCase()] ?? {
      label: color || "—",
      swatch: "#e5e7eb",
      ring: "#d6d3d1",
    }
  );
}

export interface FieldGroup {
  title: string;
  fields: { key: string; label: string }[];
}

// The substantive 29-column subset, grouped with readable RU labels for the
// SelectedTag panel. Order/keys follow CSV_COLUMNS; the six technical fields
// (filename, frame_timestamp, x/y_min/max) are intentionally excluded — they
// are not scored and are shown elsewhere (timecode, bbox overlay).
export const FIELD_GROUPS: FieldGroup[] = [
  {
    title: "Товар",
    fields: [
      { key: "product_name", label: "Наименование" },
      { key: "color", label: "Тип ценника" },
      { key: "id_sku", label: "Артикул (SKU)" },
      { key: "code", label: "Код" },
      { key: "additional_info", label: "Доп. информация" },
      { key: "special_symbols", label: "Спецсимволы" },
    ],
  },
  {
    title: "Цены",
    fields: [
      { key: "price_default", label: "Цена без карты" },
      { key: "price_card", label: "Цена по карте" },
      { key: "price_discount", label: "Цена со скидкой" },
      { key: "discount_amount", label: "Размер скидки" },
    ],
  },
  {
    title: "Идентификация",
    fields: [
      { key: "barcode", label: "Штрихкод" },
      { key: "print_datetime", label: "Дата печати" },
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
];
