import { Card } from "@/components/ui/card";
import type { JobPredictions } from "@/api/jobs";
import { colorMeta, completeness, PASS_THRESHOLD } from "@/lib/tags";

// Three at-a-glance numbers + a colour breakdown. Compact data cards
// (DESIGN: 14px label / large metric value, 24px padding).
export function Summary({ data }: { data: JobPredictions }) {
  const { tags, substantive_fields } = data;
  const scores = tags.map((t) => completeness(t, substantive_fields));
  const avg = scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : 0;
  const passing = scores.filter((s) => s >= PASS_THRESHOLD).length;

  const byColor = tags.reduce<Record<string, number>>((acc, t) => {
    acc[t.color] = (acc[t.color] ?? 0) + 1;
    return acc;
  }, {});

  return (
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
      <Stat label="Уникальных ценников" value={String(tags.length)} />
      <Stat label="Средняя полнота полей" value={`${Math.round(avg * 100)}%`} />
      <Stat
        label="Проходят порог 80%"
        value={`${passing} / ${tags.length}`}
        hint="task.md §5.1"
      />
      <Card className="p-6">
        <p className="text-[14px] text-ash-gray">Типы ценников</p>
        <div className="mt-3 flex flex-wrap gap-2">
          {Object.entries(byColor).map(([color, n]) => {
            const m = colorMeta(color);
            return (
              <span
                key={color}
                className="inline-flex items-center gap-1.5 rounded-pill border border-stone-border bg-canvas-fog px-2.5 py-1 text-[12px] text-slate-text"
              >
                <span
                  className="size-3 rounded-full"
                  style={{ background: m.swatch, boxShadow: `0 0 0 1px ${m.ring}` }}
                />
                {n}
              </span>
            );
          })}
        </div>
      </Card>
    </div>
  );
}

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <Card className="p-6">
      <p className="text-[14px] text-ash-gray">{label}</p>
      <p className="mt-2 font-display text-heading-lg font-medium text-slate-text">{value}</p>
      {hint && <p className="mt-1 text-caption text-steel-gray">{hint}</p>}
    </Card>
  );
}
