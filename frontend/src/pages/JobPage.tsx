import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  ArrowLeft,
  ChevronLeft,
  ChevronRight,
  Crop,
  Download,
  Film,
} from "lucide-react";
import {
  jobsApi,
  type Job,
  type JobPredictions,
  type TagPrediction,
} from "@/api/jobs";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { Spinner } from "@/components/ui/spinner";
import { Summary } from "@/components/job/Summary";
import { TagsTable } from "@/components/job/TagsTable";
import { TagFields } from "@/components/job/TagFields";
import { colorMeta, completeness, PASS_THRESHOLD, tagLabel } from "@/lib/tags";
import { cn, formatTimestamp } from "@/lib/utils";

const POLL_MS = 1200;

export default function JobPage() {
  const { id = "" } = useParams();
  const [job, setJob] = useState<Job | null>(null);
  const [pred, setPred] = useState<JobPredictions | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState(0);

  // --- Poll job status until it leaves queued/running -------------------
  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout>;

    const tick = async () => {
      try {
        const j = await jobsApi.get(id);
        if (!alive) return;
        setJob(j);
        if (j.status === "queued" || j.status === "running") {
          timer = setTimeout(tick, POLL_MS);
        }
      } catch {
        if (alive) setError("Задача не найдена. Возможно, бэкенд перезапустился.");
      }
    };
    tick();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, [id]);

  // --- Fetch the review payload once the job succeeds ------------------
  useEffect(() => {
    if (job?.status !== "succeeded" || pred) return;
    let alive = true;
    jobsApi
      .getPredictions(id)
      .then((p) => {
        if (!alive) return;
        setPred(p);
        setSelected(p.tags[0]?.index ?? 0);
      })
      .catch(() => alive && setError("Не удалось загрузить результаты распознавания."));
    return () => {
      alive = false;
    };
  }, [job?.status, pred, id]);

  if (error) return <ErrorCard message={error} />;
  if (job?.status === "failed")
    return <ErrorCard message={job.error || "Обработка завершилась с ошибкой."} />;
  if (!job || job.status === "queued" || job.status === "running")
    return <Processing job={job} />;
  if (!pred) return <ReviewSkeleton />;

  return (
    <div className="flex flex-col gap-8">
      <Header job={job} count={pred.tags.length} csvHref={jobsApi.csvUrl(id)} />
      <Summary data={pred} />
      <Reviewer id={id} pred={pred} selected={selected} onSelect={setSelected} />

      <Card>
        <CardHeader>
          <CardTitle>Таблица · все {pred.tags.length} ценников</CardTitle>
        </CardHeader>
        <CardContent>
          <TagsTable data={pred} selected={selected} onSelect={setSelected} />
        </CardContent>
      </Card>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Per-tag reviewer — one tag at a time. The shelf frame with *only* this tag
// highlighted (everything else dimmed), the exact crop the OCR saw, and the
// recognized fields. Step with the slider, ‹ ›, arrow keys or the filmstrip.
// ---------------------------------------------------------------------------

function Reviewer({
  id,
  pred,
  selected,
  onSelect,
}: {
  id: string;
  pred: JobPredictions;
  selected: number;
  onSelect: (index: number) => void;
}) {
  const order = pred.tags;
  const pos = Math.max(
    0,
    order.findIndex((t) => t.index === selected),
  );
  const tag = order[pos] ?? order[0];

  const go = useCallback(
    (next: number) => {
      const clamped = Math.min(order.length - 1, Math.max(0, next));
      onSelect(order[clamped].index);
    },
    [order, onSelect],
  );

  // Arrow-key stepping (ignore while typing in an input/slider).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null;
      if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA")) return;
      if (e.key === "ArrowLeft") {
        e.preventDefault();
        go(pos - 1);
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        go(pos + 1);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [go, pos]);

  const score = completeness(tag, pred.substantive_fields);
  const pass = score >= PASS_THRESHOLD;

  return (
    <Card feature>
      <div className="flex flex-col gap-1 border-b border-stone-border p-6 pb-5">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="min-w-0">
            <p className="text-caption uppercase tracking-[0.12em] text-steel-gray">
              Проверка ценников
            </p>
            <h2 className="mt-1 truncate font-display text-heading font-medium text-slate-text">
              {tagLabel(tag)}
            </h2>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => go(pos - 1)}
              disabled={pos === 0}
              title="Предыдущий (←)"
            >
              <ChevronLeft /> Назад
            </Button>
            <span className="min-w-[5.5rem] text-center font-display text-heading-sm tabular-nums text-slate-text">
              {pos + 1}{" "}
              <span className="text-ash-gray">/ {order.length}</span>
            </span>
            <Button
              variant="outline"
              size="sm"
              onClick={() => go(pos + 1)}
              disabled={pos === order.length - 1}
              title="Следующий (→)"
            >
              Вперёд <ChevronRight />
            </Button>
          </div>
        </div>
        {/* Scrubber across all tags */}
        <input
          type="range"
          min={0}
          max={order.length - 1}
          value={pos}
          onChange={(e) => go(Number(e.target.value))}
          aria-label="Перемотка по ценникам"
          className="mt-4 h-1.5 w-full cursor-pointer appearance-none rounded-pill bg-stone-border accent-chartwell-blue"
        />
      </div>

      <div className="grid gap-6 p-6 lg:grid-cols-[1.15fr_1fr]">
        <FramePanel id={id} tag={tag} />
        <div className="flex flex-col gap-4">
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant={pass ? "success" : "warning"}>
              Полнота {Math.round(score * 100)}%
            </Badge>
            <span className="text-caption text-steel-gray">
              {pass
                ? "проходит порог 80%"
                : "ниже порога 80% (task.md §5.1)"}
            </span>
          </div>
          <TagFields tag={tag} />
        </div>
      </div>

      <Filmstrip
        tags={order}
        substantive={pred.substantive_fields}
        pos={pos}
        onPick={go}
      />
    </Card>
  );
}

// The shelf frame seeked to this tag's best moment, with everything *except*
// this tag dimmed (a 9999px shadow clipped by the container) so the eye lands
// straight on it — plus the exact crop the recognizer received.
function FramePanel({ id, tag }: { id: string; tag: TagPrediction }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [ready, setReady] = useState(false);
  const [cropFailed, setCropFailed] = useState(false);

  const drawCrop = useCallback(() => {
    const v = videoRef.current;
    const c = canvasRef.current;
    if (!v || !c || !v.videoWidth || !v.videoHeight) return;
    const { x1, y1, x2, y2 } = tag.bbox;
    const sx = x1 * v.videoWidth;
    const sy = y1 * v.videoHeight;
    const sw = Math.max(1, (x2 - x1) * v.videoWidth);
    const sh = Math.max(1, (y2 - y1) * v.videoHeight);
    const W = 560;
    c.width = W;
    c.height = Math.round((W * sh) / sw);
    const ctx = c.getContext("2d");
    if (!ctx) return;
    try {
      ctx.drawImage(v, sx, sy, sw, sh, 0, 0, c.width, c.height);
      setCropFailed(false);
    } catch {
      setCropFailed(true); // cross-origin taint — only in a misconfigured deploy
    }
  }, [tag.bbox]);

  // Seek the uploaded clip to this tag's moment. The mock can't know the real
  // clip length, so it gives a fraction; multiply by the real duration
  // (frame_timestamp stays the graded ms value, shown as-is).
  useEffect(() => {
    const v = videoRef.current;
    if (!v || !ready) return;
    const dur = v.duration;
    const t = Number.isFinite(dur) ? tag.t_frac * dur : tag.frame_timestamp / 1000;
    v.currentTime = Math.min(Number.isFinite(dur) ? dur - 0.05 : t, Math.max(0, t));
  }, [tag, ready]);

  const m = colorMeta(tag.color);

  return (
    <div className="flex flex-col gap-4">
      <div>
        <p className="mb-2 flex items-center gap-2 text-caption text-ash-gray">
          <Film className="size-3.5" /> Где на полке · кадр{" "}
          {formatTimestamp(tag.frame_timestamp)}
        </p>
        <div className="relative overflow-hidden rounded-input bg-ghost-ink">
          <video
            ref={videoRef}
            src={jobsApi.videoUrl(id)}
            crossOrigin="anonymous"
            controls
            playsInline
            preload="auto"
            className="block max-h-[440px] w-full object-contain"
            onLoadedMetadata={() => setReady(true)}
            onSeeked={drawCrop}
          />
          {/* Single focus box — the 9999px box-shadow dims everything outside
              it, so exactly one tag is in question (no box soup). */}
          <div
            className="pointer-events-none absolute rounded-[3px] border-2 border-chartwell-blue"
            style={{
              left: `${tag.bbox.x1 * 100}%`,
              top: `${tag.bbox.y1 * 100}%`,
              width: `${(tag.bbox.x2 - tag.bbox.x1) * 100}%`,
              height: `${(tag.bbox.y2 - tag.bbox.y1) * 100}%`,
              boxShadow: "0 0 0 9999px rgba(15, 23, 42, 0.55)",
            }}
          />
        </div>
      </div>

      <div>
        <p className="mb-2 flex items-center gap-2 text-caption text-ash-gray">
          <Crop className="size-3.5" /> Что увидел распознаватель
        </p>
        <div className="flex flex-col items-start gap-3">
          <div className="overflow-hidden rounded-input border border-stone-border bg-canvas-fog">
            <canvas ref={canvasRef} className="block max-h-[240px] max-w-full" />
          </div>
          {cropFailed && (
            <p className="text-caption text-amber-700">
              Кроп недоступен: видео отдаётся с другого источника без CORS.
            </p>
          )}
          <div className="flex flex-wrap items-center gap-2 text-caption text-ash-gray">
            <Badge variant="info">{formatTimestamp(tag.frame_timestamp)}</Badge>
            <span>
              лучший кадр · таймкод{" "}
              <span className="font-mono text-slate-text">
                {tag.frame_timestamp}
              </span>{" "}
              мс
            </span>
            <span
              className="inline-flex items-center gap-1.5 rounded-pill border border-stone-border bg-canvas-fog px-2 py-0.5 text-slate-text"
              title={m.label}
            >
              <span
                className="size-2.5 rounded-full"
                style={{ background: m.swatch, boxShadow: `0 0 0 1px ${m.ring}` }}
              />
              {m.label}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

// Compact navigation rail — one tile per tag, tinted by its colour, ringed by
// pass/fail, current one lifted. Click jumps; horizontally scrollable.
function Filmstrip({
  tags,
  substantive,
  pos,
  onPick,
}: {
  tags: TagPrediction[];
  substantive: string[];
  pos: number;
  onPick: (pos: number) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);

  // Keep the active tile in view as you step.
  useEffect(() => {
    const el = ref.current?.children[pos] as HTMLElement | undefined;
    el?.scrollIntoView({ block: "nearest", inline: "center", behavior: "smooth" });
  }, [pos]);

  const tiles = useMemo(
    () =>
      tags.map((t, i) => {
        const m = colorMeta(t.color);
        const pass = completeness(t, substantive) >= PASS_THRESHOLD;
        return { t, i, m, pass };
      }),
    [tags, substantive],
  );

  return (
    <div
      ref={ref}
      className="flex gap-2 overflow-x-auto border-t border-stone-border p-4"
    >
      {tiles.map(({ t, i, m, pass }) => {
        const active = i === pos;
        return (
          <button
            key={t.index}
            type="button"
            onClick={() => onPick(i)}
            title={tagLabel(t)}
            className={cn(
              "relative flex shrink-0 flex-col items-center gap-1 rounded-input border px-3 py-2 transition-colors",
              active
                ? "border-chartwell-blue bg-chartwell-blue/10"
                : "border-stone-border bg-cloud-white hover:bg-canvas-fog",
            )}
          >
            <span
              className="size-3 rounded-full"
              style={{ background: m.swatch, boxShadow: `0 0 0 1px ${m.ring}` }}
            />
            <span
              className={cn(
                "text-[12px] tabular-nums",
                active ? "text-chartwell-blue" : "text-ash-gray",
              )}
            >
              {i + 1}
            </span>
            <span
              className={cn(
                "h-1 w-6 rounded-pill",
                pass ? "bg-chartwell-blue" : "bg-amber-400",
              )}
            />
          </button>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------

function Header({
  job,
  count,
  csvHref,
}: {
  job: Job;
  count: number;
  csvHref: string;
}) {
  return (
    <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
      <div className="min-w-0">
        <Link
          to="/"
          className="inline-flex items-center gap-1.5 text-caption text-ash-gray hover:text-slate-text"
        >
          <ArrowLeft className="size-3.5" /> Новое видео
        </Link>
        <h1 className="mt-2 truncate font-display text-heading-lg font-medium text-slate-text">
          {job.filename}
        </h1>
        <div className="mt-2 flex items-center gap-2">
          <Badge variant="success">Готово</Badge>
          <span className="text-caption text-ash-gray">
            {count} уникальных ценников
          </span>
        </div>
      </div>
      <a href={csvHref} download>
        <Button size="lg">
          <Download />
          Скачать CSV
        </Button>
      </a>
    </div>
  );
}

function Processing({ job }: { job: Job | null }) {
  const progress = job?.progress ?? 0;
  const phase =
    !job || job.status === "queued"
      ? "В очереди…"
      : progress < 0.5
        ? "Детекция ценников в кадрах…"
        : progress < 0.9
          ? "Распознавание полей и штрихкодов…"
          : "Сборка выгрузки…";
  return (
    <div className="grid place-items-center py-24">
      <Card feature className="w-full max-w-md p-8 text-center">
        <div className="mx-auto grid size-12 place-items-center rounded-card bg-chartwell-blue/10 text-chartwell-blue">
          <Spinner className="size-6" />
        </div>
        <h1 className="mt-5 font-display text-heading font-medium text-slate-text">
          Обрабатываем видео
        </h1>
        {job && (
          <p className="mt-1 truncate text-caption text-ash-gray">{job.filename}</p>
        )}
        <div className="mt-6">
          <Progress value={progress} />
          <div className="mt-2 flex justify-between text-caption text-ash-gray">
            <span>{phase}</span>
            <span className="tabular-nums">{Math.round(progress * 100)}%</span>
          </div>
        </div>
        <p className="mt-6 text-caption text-steel-gray">
          Видео обрабатывается покадрово — это занимает время. Экран обновится
          автоматически.
        </p>
      </Card>
    </div>
  );
}

function ReviewSkeleton() {
  return (
    <div className="flex flex-col gap-8">
      <Skeleton className="h-12 w-72" />
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <Skeleton key={i} className="h-28" />
        ))}
      </div>
      <Skeleton className="h-[520px]" />
    </div>
  );
}

function ErrorCard({ message }: { message: string }) {
  return (
    <div className="grid place-items-center py-24">
      <Card className="w-full max-w-md p-8 text-center">
        <h1 className="font-display text-heading font-medium text-slate-text">
          Не получилось
        </h1>
        <p className="mt-2 text-[14px] text-ash-gray">{message}</p>
        <Link to="/">
          <Button className="mt-6">Загрузить другое видео</Button>
        </Link>
      </Card>
    </div>
  );
}
