import type { TagPrediction } from "@/api/jobs";
import { colorMeta, FIELD_GROUPS, fieldState } from "@/lib/tags";
import { TAG_SCHEMA_COPY } from "@/content/tagSchemaContent";

// The recognized data, grouped and labelled — the "красиво распознанные
// данные" the review screen exists for. The three task.md §3.3 states are
// visually distinct: a value, "нет на ценнике" (absent), and "не
// распознано" (present but unread) — confusing the last two loses points
// (task.md §5.3), so they must never look the same.
export function TagFields({ tag }: { tag: TagPrediction }) {
  return (
    <div className="flex flex-col gap-6">
      {FIELD_GROUPS.map((group) => (
        <section key={group.title}>
          <h4 className="mb-2 text-caption font-medium uppercase tracking-[0.12em] text-steel-gray">
            {group.title}
          </h4>
          <dl className="divide-y divide-stone-border/70 overflow-hidden rounded-input border border-stone-border">
            {group.fields.map(({ key, label }) => (
              <div
                key={key}
                className="flex items-baseline justify-between gap-4 bg-cloud-white px-3 py-2"
              >
                <dt className="shrink-0 text-[13px] text-ash-gray">{label}</dt>
                <dd className="min-w-0 text-right">
                  <FieldValue field={key} value={tag.fields[key]} color={tag.color} />
                </dd>
              </div>
            ))}
          </dl>
        </section>
      ))}
    </div>
  );
}

function FieldValue({
  field,
  value,
  color,
}: {
  field: string;
  value: string | undefined;
  color: string;
}) {
  const state = fieldState(value);

  if (state === "absent") {
    return (
      <span className="inline-flex items-center rounded-pill border border-stone-border bg-canvas-fog px-2 py-0.5 text-[12px] text-ash-gray">
        {TAG_SCHEMA_COPY.tagFields.absent}
      </span>
    );
  }
  if (state === "unrecognized") {
    return (
      <span className="inline-flex items-center rounded-pill border border-amber-200 bg-amber-50 px-2 py-0.5 text-[12px] text-amber-700">
        {TAG_SCHEMA_COPY.tagFields.unrecognized}
      </span>
    );
  }
  if (field === "color") {
    const m = colorMeta(color);
    return (
      <span className="inline-flex items-center gap-1.5 text-[13px] text-slate-text">
        <span
          className="size-3 rounded-full"
          style={{ background: m.swatch, boxShadow: `0 0 0 1px ${m.ring}` }}
        />
        {m.label}
      </span>
    );
  }
  const mono = field === "barcode" || field.includes("qr") || field.startsWith("x_") || field.startsWith("y_");
  return (
    <span
      className={`break-words text-[13px] text-slate-text ${mono ? "font-mono tabular-nums" : ""}`}
    >
      {value}
    </span>
  );
}
