// Per-tag presentation logic shared by the review screen. Mirrors the
// graded contract in backend/app/api/v1/schemas/job.py (CSV_COLUMNS /
// SUBSTANTIVE_FIELDS) and the three task.md §3.3 field states.
//
// (Tracked source — see the note in ./utils.ts about the .gitignore fix.)
import { ABSENT, type TagPrediction } from "@/api/jobs";
import {
  TAG_SCHEMA_COPY,
  type ColorMetaCopy,
  type FieldGroupCopy,
} from "@/content/tagSchemaContent";

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
    : `${TAG_SCHEMA_COPY.fallbackTagPrefix}${tag.index + 1}`;
}

export interface ColorMeta {
  label: string;
  swatch: string; // fill
  ring: string; // 1px outline so a near-white chip stays visible
}

// The tag colour encodes the price mechanic. The reviewer is for store
// staff, not us — show the plain meaning, never the internal colour name.
const COLOR_META: Record<string, ColorMetaCopy> = TAG_SCHEMA_COPY.colorMeta;

export function colorMeta(color: string): ColorMeta {
  return (
    COLOR_META[(color || "").toLowerCase()] ?? {
      label: color || "—",
      swatch: "#e5e7eb",
      ring: "#d6d3d1",
    }
  );
}

// task.md §3: `special_symbols` is the tag's *display type*, not "special
// symbols" — к (коробка), л (лоток), ш (штука). Shown human-readably so the
// reviewer doesn't see a bare "к".
const DISPLAY_TYPE: Record<string, string> = {
  ...TAG_SCHEMA_COPY.displayType,
};

/** Human-readable cell value for a field (currently only the display type
 *  needs decoding; everything else is shown verbatim). */
export function displayValue(field: string, value: string): string {
  if (field === "special_symbols") return DISPLAY_TYPE[value] ?? value;
  return value;
}

export interface FieldGroup {
  title: string;
  fields: { key: string; label: string }[];
}

// The substantive 29-column subset, grouped with readable RU labels for the
// SelectedTag panel. Order/keys follow CSV_COLUMNS; the six technical fields
// (filename, frame_timestamp, x/y_min/max) are intentionally excluded — they
// are not scored and are shown elsewhere (timecode, bbox overlay).
export const FIELD_GROUPS: FieldGroup[] = TAG_SCHEMA_COPY.fieldGroups as FieldGroupCopy[];
