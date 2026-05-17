import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ArrowLeft, Crop, Download, Film } from "lucide-react";
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

  const tag = pred.tags.find((t) => t.index === selected) ?? pred.tags[0];

  return (
    <div className="flex flex-col gap-8">
      <Header job={job} count={pred.tags.length} csvHref={jobsApi.csvUrl(id)} />
      <Summary data={pred} />

      <div className="grid gap-6 lg:grid-cols-2">
        <VideoPanel id={id} pred={pred} tag={tag} onSelect={setSelected} />
        <SelectedTag pred={pred} tag={tag} />
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Все ценники · {pred.tags.length}</CardTitle>
        </CardHeader>
        <CardContent>
          <TagsTable data={pred} selected={selected} onSelect={setSelected} />
        </CardContent>
      </Card>
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
          <span className="text-caption text-ash-gray">{count} уникальных ценников</span>
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

function VideoPanel({
  id,
  pred,
  tag,
  onSelect,
}: {
  id: string;
  pred: JobPredictions;
  tag: TagPrediction;
  onSelect: (index: number) => void;
}) {
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

  // Seek the uploaded clip to this tag's moment. The mock can't know the
  // real clip length, so it gives a fraction; multiply by the actual
  // duration (frame_timestamp stays the graded ms value, shown as-is).
  useEffect(() => {
    const v = videoRef.current;
    if (!v || !ready) return;
    const dur = v.duration;
    const t = Number.isFinite(dur) ? tag.t_frac * dur : tag.frame_timestamp / 1000;
    v.currentTime = Math.min(Number.isFinite(dur) ? dur - 0.05 : t, Math.max(0, t));
  }, [tag, ready]);

  const m = colorMeta(tag.color);

  return (
    <div className="flex flex-col gap-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Film className="size-4 text-ash-gray" /> Исходное видео
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="relative overflow-hidden rounded-input bg-ghost-ink">
            <video
              ref={videoRef}
              src={jobsApi.videoUrl(id)}
              crossOrigin="anonymous"
              controls
              playsInline
              preload="auto"
              className="block max-h-[420px] w-full object-contain"
              onLoadedMetadata={() => setReady(true)}
              onSeeked={drawCrop}
            />
            {/* Bounding boxes overlaid in normalized coords — scales with the
                rendered <video> automatically. */}
            {pred.tags.map((t) => {
              const active = t.index === tag.index;
              return (
                <button
                  key={t.index}
                  type="button"
                  onClick={() => onSelect(t.index)}
                  title={tagLabel(t)}
                  className={cn(
                    "absolute rounded-[3px] transition-colors",
                    active
                      ? "border-2 border-chartwell-blue bg-chartwell-blue/10"
                      : "border border-cloud-white/40 hover:border-cloud-white",
                  )}
                  style={{
                    left: `${t.bbox.x1 * 100}%`,
                    top: `${t.bbox.y1 * 100}%`,
                    width: `${(t.bbox.x2 - t.bbox.x1) * 100}%`,
                    height: `${(t.bbox.y2 - t.bbox.y1) * 100}%`,
                  }}
                >
                  {active && (
                    <span className="absolute -top-5 left-0 whitespace-nowrap rounded-pill bg-chartwell-blue px-1.5 py-0.5 text-[10px] font-medium text-cloud-white">
                      #{t.index + 1}
                    </span>
                  )}
                </button>
              );
            })}
          </div>
          <p className="mt-2 text-caption text-ash-gray">
            Прямоугольники — найденные ценники. Клик по рамке или строке
            таблицы переключает выбранный ценник.
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Crop className="size-4 text-ash-gray" /> Кроп ценника #{tag.index + 1}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col items-start gap-3">
            <div className="overflow-hidden rounded-input border border-stone-border bg-canvas-fog">
              <canvas ref={canvasRef} className="block max-w-full" />
            </div>
            {cropFailed && (
              <p className="text-caption text-amber-700">
                Кроп недоступен: видео отдаётся с другого источника без CORS.
              </p>
            )}
            <div className="flex flex-wrap items-center gap-2 text-caption text-ash-gray">
              <Badge variant="info">{formatTimestamp(tag.frame_timestamp)}</Badge>
              <span>
                кадр, где ценник распознан лучше всего · таймкод{" "}
                <span className="font-mono text-slate-text">{tag.frame_timestamp}</span> мс
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
        </CardContent>
      </Card>
    </div>
  );
}

function SelectedTag({ pred, tag }: { pred: JobPredictions; tag: TagPrediction }) {
  const score = completeness(tag, pred.substantive_fields);
  const pass = score >= PASS_THRESHOLD;
  return (
    <Card>
      <CardHeader>
        <CardTitle className="truncate">{tagLabel(tag)}</CardTitle>
        <div className="mt-1 flex flex-wrap items-center gap-2">
          <Badge variant={pass ? "success" : "warning"}>
            Полнота {Math.round(score * 100)}%
          </Badge>
          <span className="text-caption text-steel-gray">
            {pass ? "проходит порог 80%" : "ниже порога 80% (task.md §5.1)"}
          </span>
        </div>
      </CardHeader>
      <CardContent>
        <TagFields tag={tag} />
      </CardContent>
    </Card>
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
      <div className="grid gap-6 lg:grid-cols-2">
        <Skeleton className="h-96" />
        <Skeleton className="h-96" />
      </div>
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
