import { useState } from "react";
import type { LucideIcon } from "lucide-react";
import {
  Check,
  ExternalLink,
  FlaskConical,
  Languages,
  QrCode,
  ScanSearch,
  SlidersHorizontal,
  Sparkles,
  X,
} from "lucide-react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import {
  EXPERIMENTS_PAGE_COPY,
  EXPERIMENTS_SECTIONS_COPY,
  type Verdict,
} from "@/content/experimentsContent";

const SECTION_ICONS: Record<string, LucideIcon> = {
  detector: ScanSearch,
  ocr: Languages,
  qr: QrCode,
  tuning: SlidersHorizontal,
};

const VERDICT: Record<
  Verdict,
  { label: string; hint: string; icon: LucideIcon; cls: string }
> = {
  prod: {
    ...EXPERIMENTS_PAGE_COPY.verdict.prod,
    icon: Check,
    cls: "border-chartwell-blue/40 bg-chartwell-blue/10 text-slate-text",
  },
  explored: {
    ...EXPERIMENTS_PAGE_COPY.verdict.explored,
    icon: FlaskConical,
    cls: "border-stone-border bg-canvas-fog text-ash-gray",
  },
  rejected: {
    ...EXPERIMENTS_PAGE_COPY.verdict.rejected,
    icon: X,
    cls: "border-stone-border bg-transparent text-steel-gray",
  },
};

const VERDICT_ORDER: Verdict[] = ["prod", "explored", "rejected"];

export default function ExperimentsPage() {
  const [activeId, setActiveId] = useState(EXPERIMENTS_SECTIONS_COPY[0].id);
  const section =
    EXPERIMENTS_SECTIONS_COPY.find((s) => s.id === activeId) ??
    EXPERIMENTS_SECTIONS_COPY[0];
  const ActiveIcon = SECTION_ICONS[section.id] ?? ScanSearch;

  return (
    <div className="flex flex-col gap-10">
      <section className="mx-auto max-w-2xl pt-6 text-center">
        <p className="text-caption font-medium uppercase tracking-[0.14em] text-chartwell-blue">
          {EXPERIMENTS_PAGE_COPY.heroKicker}
        </p>
        <h1 className="mt-4 font-display text-heading-lg font-medium text-slate-text sm:text-display">
          {EXPERIMENTS_PAGE_COPY.heroTitle}
        </h1>
        <p className="mx-auto mt-4 max-w-xl text-[16px] leading-[1.6] text-ash-gray">
          {EXPERIMENTS_PAGE_COPY.heroLead}
        </p>
        <div className="mt-6 flex flex-wrap items-center justify-center gap-2">
          <Badge variant="accent">
            <Sparkles className="size-3" /> {EXPERIMENTS_PAGE_COPY.heroBadges[0]}
          </Badge>
          <Badge variant="neutral">{EXPERIMENTS_PAGE_COPY.heroBadges[1]}</Badge>
        </div>
      </section>

      <section className="mx-auto w-full max-w-4xl">
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          {EXPERIMENTS_SECTIONS_COPY.map((s) => {
            const Icon = SECTION_ICONS[s.id] ?? ScanSearch;
            const on = s.id === activeId;
            return (
              <button
                key={s.id}
                type="button"
                onClick={() => setActiveId(s.id)}
                aria-pressed={on}
                className={cn(
                  "flex cursor-pointer items-center gap-2 rounded-card border px-4 py-3 text-left transition-all duration-200",
                  on
                    ? "border-chartwell-blue/40 bg-sky-tint/15 shadow-card"
                    : "border-stone-border bg-card hover:-translate-y-0.5 hover:border-chartwell-blue/40 hover:shadow-pop",
                )}
              >
                <span
                  className={cn(
                    "grid size-8 shrink-0 place-items-center rounded-input transition-colors",
                    on
                      ? "bg-chartwell-blue text-cloud-white"
                      : "bg-chartwell-blue/10 text-chartwell-blue",
                  )}
                >
                  <Icon className="size-4" />
                </span>
                <span className="min-w-0">
                  <span className="block truncate text-[14px] font-medium text-slate-text">
                    {s.title}
                  </span>
                  <span className="block truncate text-[11px] text-ash-gray">
                    {s.stat}
                  </span>
                </span>
              </button>
            );
          })}
        </div>

        <div className="mt-3 flex flex-wrap items-center justify-center gap-x-5 gap-y-1.5">
          {VERDICT_ORDER.map((v) => {
            const meta = VERDICT[v];
            const MIcon = meta.icon;
            return (
              <span key={v} className="inline-flex items-center gap-1.5 text-[12px] text-ash-gray">
                <span
                  className={cn(
                    "inline-flex items-center gap-1 rounded-pill border px-2 py-0.5 text-[11px] font-medium",
                    meta.cls,
                  )}
                >
                  <MIcon className="size-3" />
                  {meta.label}
                </span>
                <span className="text-steel-gray">— {meta.hint}</span>
              </span>
            );
          })}
        </div>
      </section>

      <section className="mx-auto w-full max-w-4xl">
        <Card feature className="p-6 sm:p-8">
          <div className="flex items-start gap-3">
            <span className="grid size-11 shrink-0 place-items-center rounded-card bg-chartwell-blue/10 text-chartwell-blue">
              <ActiveIcon className="size-5" />
            </span>
            <div className="min-w-0">
              <h2 className="font-display text-heading font-medium text-slate-text">
                {section.title}
              </h2>
              <p className="mt-1 text-[14px] leading-[1.6] text-ash-gray">
                {section.blurb}
              </p>
            </div>
          </div>

          {section.datasets && (
            <div className="mt-6">
              <p className="text-caption font-medium uppercase tracking-[0.12em] text-steel-gray">
                {EXPERIMENTS_PAGE_COPY.datasetsTitle}
              </p>
              <div className="mt-2 flex flex-wrap gap-2">
                {section.datasets.map((d) => (
                  <a
                    key={d.url}
                    href={d.url}
                    target="_blank"
                    rel="noreferrer noopener"
                    className="inline-flex items-center gap-1.5 rounded-pill border border-stone-border bg-canvas-fog px-3 py-1 text-[12px] text-slate-text transition-colors hover:border-chartwell-blue/40 hover:text-chartwell-blue"
                  >
                    {d.name}
                    <ExternalLink className="size-3" />
                  </a>
                ))}
              </div>
            </div>
          )}

          <div className="mt-6">
            <p className="text-caption font-medium uppercase tracking-[0.12em] text-steel-gray">
              {EXPERIMENTS_PAGE_COPY.triedTitle}
            </p>
            {section.id === "tuning" && (
              <p className="mt-2 rounded-input border border-chartwell-blue/25 bg-chartwell-blue/5 px-3 py-2 text-[12px] leading-[1.55] text-ash-gray">
                {EXPERIMENTS_PAGE_COPY.tuningNote}
              </p>
            )}
            <ul className="mt-3 flex flex-col gap-2">
              {section.tried.map((a) => {
                const v = VERDICT[a.verdict];
                const VIcon = v.icon;
                return (
                  <li
                    key={a.label}
                    className="flex items-start justify-between gap-3 rounded-input border border-stone-border bg-cloud-white px-3 py-2.5"
                  >
                    <div className="min-w-0">
                      <span
                        className={cn(
                          "text-[13px] text-slate-text",
                          a.verdict === "rejected" && "text-steel-gray",
                        )}
                      >
                        {a.label}
                      </span>
                      {a.note && (
                        <span className="ml-2 text-[12px] text-ash-gray">— {a.note}</span>
                      )}
                    </div>
                    <span
                      className={cn(
                        "inline-flex shrink-0 items-center gap-1 rounded-pill border px-2 py-0.5 text-[11px] font-medium",
                        v.cls,
                      )}
                    >
                      <VIcon className="size-3" />
                      {v.label}
                    </span>
                  </li>
                );
              })}
            </ul>
          </div>

          <div className="mt-6 grid gap-3 sm:grid-cols-2">
            <div className="rounded-card border border-chartwell-blue/30 bg-chartwell-blue/5 p-4">
              <p className="text-caption font-medium uppercase tracking-[0.12em] text-chartwell-blue">
                {EXPERIMENTS_PAGE_COPY.summaryCards.finding}
              </p>
              <p className="mt-1.5 text-[13px] leading-[1.6] text-slate-text">
                {section.finding}
              </p>
            </div>
            <div className="rounded-card border border-stone-border bg-canvas-fog p-4">
              <p className="text-caption font-medium uppercase tracking-[0.12em] text-steel-gray">
                {EXPERIMENTS_PAGE_COPY.summaryCards.shipped}
              </p>
              <p className="mt-1.5 text-[13px] leading-[1.6] text-slate-text">
                {section.shipped}
              </p>
            </div>
          </div>
        </Card>
      </section>

      <section className="mx-auto w-full max-w-4xl">
        <Card className="flex flex-col items-center gap-2 p-6 text-center">
          <p className="text-caption font-medium uppercase tracking-[0.12em] text-chartwell-blue">
            {EXPERIMENTS_PAGE_COPY.methodTitle}
          </p>
          <p className="max-w-2xl text-[14px] leading-[1.65] text-ash-gray">
            {EXPERIMENTS_PAGE_COPY.methodText}
          </p>
        </Card>
      </section>
    </div>
  );
}
