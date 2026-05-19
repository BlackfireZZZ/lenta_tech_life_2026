import { Card } from "@/components/ui/card";
import type { JobPredictions } from "@/api/jobs";
import { colorMeta } from "@/lib/tags";

// Two things a store reviewer actually cares about: how many price tags were
// found, and how they split by price type (обычная / по карте / промо /
// акция). No internal metrics — that's noise to the user.
export function Summary({ data }: { data: JobPredictions }) {
  const { tags } = data;
  const byType = tags.reduce<Record<string, number>>((acc, t) => {
    acc[t.color] = (acc[t.color] ?? 0) + 1;
    return acc;
  }, {});

  return (
    <div className="grid gap-4 sm:grid-cols-3">
      <Card className="p-6">
        <p className="text-[14px] text-ash-gray">Найдено ценников</p>
        <p className="mt-2 font-display text-heading-lg font-medium text-slate-text">
          {tags.length}
        </p>
      </Card>
      <Card className="p-6 sm:col-span-2">
        <p className="text-[14px] text-ash-gray">По типу цены</p>
        <div className="mt-3 flex flex-wrap gap-2">
          {Object.entries(byType).map(([color, n]) => {
            const m = colorMeta(color);
            return (
              <span
                key={color}
                className="inline-flex items-center gap-2 rounded-pill border border-stone-border bg-canvas-fog px-3 py-1 text-[13px] text-slate-text"
              >
                <span
                  className="size-2.5 rounded-full"
                  style={{ background: m.swatch, boxShadow: `0 0 0 1px ${m.ring}` }}
                />
                {m.label}
                <span className="tabular-nums text-ash-gray">{n}</span>
              </span>
            );
          })}
        </div>
      </Card>
    </div>
  );
}
