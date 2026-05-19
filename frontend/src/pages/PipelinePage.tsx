import { useCallback, useEffect, useRef, useState } from "react";
import type { LucideIcon } from "lucide-react";
import {
  Aperture,
  Boxes,
  ChevronDown,
  Crop,
  FileCheck2,
  Languages,
  Library,
  ListChecks,
  Pause,
  Play,
  QrCode,
  RotateCcw,
  ScanSearch,
  Sparkles,
  Spline,
  UploadCloud,
} from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import {
  PIPELINE_PAGE_COPY,
  PIPELINE_STAGES_TEXT,
  type PipelineStageText,
} from "@/content/pipelineContent";

// ─────────────────────────────────────────────────────────────────────────
// The real pipeline, stage by stage. Faithful to docs/pipeline-reference.md,
// docs/architecture.md, configs/balanced.yaml and catalog-reconciliation.md
// — this page is the jury's guided tour of how one shelf video becomes one
// graded CSV.
// ─────────────────────────────────────────────────────────────────────────
interface Stage {
  icon: LucideIcon;
  title: PipelineStageText["title"];
  tagline: PipelineStageText["tagline"];
  detail: PipelineStageText["detail"];
  techniques: PipelineStageText["techniques"];
  knobs?: PipelineStageText["knobs"];
  ref?: PipelineStageText["ref"];
}

const STAGE_ICONS: LucideIcon[] = [
  UploadCloud,
  ScanSearch,
  Spline,
  Aperture,
  Crop,
  QrCode,
  Languages,
  ListChecks,
  Boxes,
  Library,
  FileCheck2,
];

const STAGES: Stage[] = PIPELINE_STAGES_TEXT.map((stage, i) => ({
  ...stage,
  icon: STAGE_ICONS[i] ?? FileCheck2,
}));

const STEP_MS = 1700;

export default function PipelinePage() {
  const [active, setActive] = useState(0);
  const [playing, setPlaying] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  const last = STAGES.length - 1;
  const pct = (active / last) * 100;

  // Auto-advance while playing; stop at the final stage.
  useEffect(() => {
    if (!playing) return;
    if (active >= last) {
      setPlaying(false);
      return;
    }
    timer.current = setTimeout(() => setActive((i) => i + 1), STEP_MS);
    return () => clearTimeout(timer.current);
  }, [playing, active, last]);

  const onPlay = useCallback(() => {
    if (active >= last) setActive(0);
    setPlaying((p) => !p);
  }, [active, last]);

  const reset = useCallback(() => {
    setPlaying(false);
    setActive(0);
  }, []);

  const select = useCallback((i: number) => {
    setPlaying(false);
    setActive((cur) => (cur === i ? -1 : i)); // click again to collapse
  }, []);

  return (
    <div className="flex flex-col gap-12">
      <style>{`
        @keyframes plg-glow {
          0%,100% { box-shadow: 0 0 0 0 rgba(59,166,241,.45), 0 0 10px 2px rgba(59,166,241,.35); }
          50%     { box-shadow: 0 0 0 6px rgba(59,166,241,0), 0 0 18px 5px rgba(59,166,241,.55); }
        }
        .plg-head { animation: plg-glow 1.6s ease-in-out infinite; }
        @media (prefers-reduced-motion: reduce) { .plg-head { animation: none; } }
      `}</style>

      {/* Hero */}
      <section className="mx-auto max-w-2xl pt-6 text-center">
        <p className="text-caption font-medium uppercase tracking-[0.14em] text-chartwell-blue">
          {PIPELINE_PAGE_COPY.heroKicker}
        </p>
        <h1 className="mt-4 font-display text-heading-lg font-medium text-slate-text sm:text-display">
          {PIPELINE_PAGE_COPY.heroTitle}
        </h1>
        <p className="mx-auto mt-4 max-w-xl text-[16px] leading-[1.6] text-ash-gray">
          {PIPELINE_PAGE_COPY.heroLead}
        </p>
        <div className="mt-6 flex flex-wrap items-center justify-center gap-3">
          <Button
            size="lg"
            onClick={onPlay}
            aria-label={PIPELINE_PAGE_COPY.cta.runAria}
            className="min-w-[220px] shadow-card transition-transform duration-200 hover:-translate-y-0.5"
          >
            {playing ? <Pause /> : <Play />}
            {playing
              ? PIPELINE_PAGE_COPY.cta.pause
              : active >= last
                ? PIPELINE_PAGE_COPY.cta.replay
                : active <= 0
                  ? PIPELINE_PAGE_COPY.cta.start
                  : PIPELINE_PAGE_COPY.cta.resume}
          </Button>
          <Button
            variant="ghost"
            size="lg"
            onClick={reset}
            aria-label={PIPELINE_PAGE_COPY.cta.resetAria}
            className="min-w-[170px]"
          >
            <RotateCcw /> {PIPELINE_PAGE_COPY.cta.reset}
          </Button>
        </div>
        <div className="mt-6 flex flex-wrap items-center justify-center gap-2">
          <Badge variant="accent">
            <Sparkles className="size-3" /> {PIPELINE_PAGE_COPY.badges[0]}
          </Badge>
          <Badge variant="info">{PIPELINE_PAGE_COPY.badges[1]}</Badge>
          <Badge variant="neutral">{PIPELINE_PAGE_COPY.badges[2]}</Badge>
        </div>
      </section>

      {/* The flow graph */}
      <section className="relative mx-auto w-full max-w-3xl">
        {/* spine track + animated fill */}
        <div
          aria-hidden
          className="absolute bottom-3 top-3 w-px bg-stone-border left-[19px] sm:left-[23px]"
        />
        <div
          aria-hidden
          className="absolute top-3 w-px bg-chartwell-blue transition-[height] duration-700 ease-out left-[19px] sm:left-[23px]"
          style={{ height: `calc(${Math.max(0, pct)}% * 0.94)` }}
        >
          <span className="plg-head absolute -bottom-1 left-1/2 size-2.5 -translate-x-1/2 rounded-full bg-chartwell-blue" />
        </div>

        <ol className="flex flex-col gap-3">
          {STAGES.map((stage, i) => {
            const Icon = stage.icon;
            const done = active >= 0 && i < active;
            const isActive = i === active;
            return (
              <li key={stage.title} className="relative pl-12 sm:pl-16">
                {/* node bullet on the spine */}
                <span
                  className={cn(
                    "absolute top-4 grid size-10 place-items-center rounded-full border-2 bg-cloud-white transition-colors duration-300 left-0 sm:size-12",
                    isActive
                      ? "border-chartwell-blue text-chartwell-blue shadow-pop"
                      : done
                        ? "border-chartwell-blue/40 text-chartwell-blue/70"
                        : "border-stone-border text-steel-gray",
                  )}
                >
                  <Icon className="size-4 sm:size-5" />
                </span>

                <button
                  type="button"
                  onClick={() => select(i)}
                  aria-expanded={isActive}
                  className="group block w-full cursor-pointer text-left"
                >
                  <Card
                    feature={isActive}
                    className={cn(
                      "p-5 transition-all duration-300 sm:p-6",
                      "group-focus-visible:ring-2 group-focus-visible:ring-chartwell-blue/40",
                      isActive
                        ? "border-chartwell-blue/40 bg-sky-tint/15"
                        : "group-hover:-translate-y-0.5 group-hover:border-chartwell-blue/40 group-hover:bg-canvas-fog group-hover:shadow-pop",
                    )}
                  >
                    <div className="flex items-center justify-between gap-3">
                      <div className="min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="text-caption font-medium tabular-nums text-steel-gray">
                            {String(i + 1).padStart(2, "0")}
                          </span>
                          <h3 className="truncate font-display text-heading-sm font-medium text-slate-text">
                            {stage.title}
                          </h3>
                        </div>
                        <p className="mt-0.5 truncate text-[13px] text-ash-gray">
                          {stage.tagline}
                        </p>
                      </div>

                      {/* Clear "this is clickable" affordance */}
                      <div className="flex shrink-0 items-center gap-2">
                        {stage.ref && (
                          <span className="hidden rounded-pill border border-stone-border bg-canvas-fog px-2.5 py-0.5 font-mono text-[11px] text-steel-gray md:block">
                            {stage.ref}
                          </span>
                        )}
                        <span
                          className={cn(
                            "hidden text-caption font-medium transition-colors sm:block",
                            isActive
                              ? "text-chartwell-blue"
                              : "text-steel-gray group-hover:text-chartwell-blue",
                          )}
                        >
                          {isActive
                            ? PIPELINE_PAGE_COPY.stageToggle.collapse
                            : PIPELINE_PAGE_COPY.stageToggle.expand}
                        </span>
                        <span
                          className={cn(
                            "grid size-7 place-items-center rounded-pill border transition-all duration-300",
                            isActive
                              ? "rotate-180 border-chartwell-blue/40 bg-chartwell-blue/10 text-chartwell-blue"
                              : "border-stone-border text-steel-gray group-hover:border-chartwell-blue/40 group-hover:text-chartwell-blue",
                          )}
                        >
                          <ChevronDown className="size-4" />
                        </span>
                      </div>
                    </div>

                    {/* Expanded detail — only for the active stage */}
                    <div
                      className={cn(
                        "grid transition-all duration-300",
                        isActive
                          ? "mt-4 grid-rows-[1fr] opacity-100"
                          : "grid-rows-[0fr] opacity-0",
                      )}
                    >
                      <div className="overflow-hidden">
                        <p className="text-[14px] leading-[1.65] text-slate-text">
                          {stage.detail}
                        </p>
                        <div className="mt-4 flex flex-wrap gap-2">
                          {stage.techniques.map((t) => (
                            <span
                              key={t}
                              className="inline-flex items-center gap-1.5 rounded-pill border border-chartwell-blue/30 bg-chartwell-blue/10 px-2.5 py-0.5 text-[12px] font-medium text-slate-text"
                            >
                              {t}
                            </span>
                          ))}
                        </div>
                        {stage.knobs && (
                          <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1">
                            {stage.knobs.map((k) => (
                              <code
                                key={k}
                                className="font-mono text-[12px] text-ash-gray"
                              >
                                {k}
                              </code>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  </Card>
                </button>
              </li>
            );
          })}
        </ol>
      </section>

      {/* Closing strip */}
      <section className="mx-auto w-full max-w-3xl">
        <Card className="flex flex-col items-center gap-2 p-6 text-center">
          <p className="text-caption font-medium uppercase tracking-[0.12em] text-chartwell-blue">
            {PIPELINE_PAGE_COPY.closing.title}
          </p>
          <p className="max-w-xl text-[14px] leading-[1.65] text-ash-gray">
            {PIPELINE_PAGE_COPY.closing.text}
          </p>
        </Card>
      </section>
    </div>
  );
}
