import {
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Link, useParams } from "react-router-dom";
import {
  ArrowLeft,
  ChevronLeft,
  ChevronRight,
  Clock,
  Crop,
  Download,
  Film,
  Image as ImageIcon,
  Pause,
  Play,
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
import { ServerLimitNote } from "@/components/ServerLimitNote";
import {
  colorMeta,
  completeness,
  displayValue,
  FIELD_GROUPS,
  fieldState,
  PASS_THRESHOLD,
  tagLabel,
} from "@/lib/tags";
import { cn } from "@/lib/utils";

const POLL_MS = 1200;

// Tags whose best-frame timestamps fall within this window are treated as
// "the same frame" (so a busy shelf moment is one group of many boxes).
const FRAME_BUCKET_MS = 200;

// Live ("видео с разметкой") view: a tag has a single captured moment — one
// CSV row = one best-frame timestamp; the graded contract carries no
// per-frame track. So as the clip plays we light its box for a short window
// AROUND that moment, just wide enough to be seen at normal speed. This
// marks *when the tag was read*, it does not claim a continuous track.
const LIVE_WINDOW_S = 0.6;

// The two honest review modes (see ViewModeBar).
type ViewMode = "frame" | "live";

// Shown big in the hero strip — everything else folds into the detail
// groups, so the panel is never one endless column (the rest of the
// FIELD_GROUPS rows skip these keys to avoid repetition).
const HERO_KEYS = [
  "price_default",
  "price_card",
  "barcode",
  "id_sku",
] as const;
const HIDDEN_FROM_GROUPS = new Set<string>([
  "product_name",
  "color",
  ...HERO_KEYS,
]);

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
        if (alive) setError("Не удалось найти эту обработку — возможно, сервер перезапустился.");
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

  // Only surface tags we're confident in — a half-read ценник is noise to a
  // reviewer, not data. The downloaded CSV still contains every row (it is
  // the deliverable); the on-screen review just hides the unreliable ones.
  const view = useMemo(() => {
    if (!pred) return null;
    const good = pred.tags.filter(
      (t) => completeness(t, pred.substantive_fields) >= PASS_THRESHOLD,
    );
    return { ...pred, tags: good.length ? good : pred.tags };
  }, [pred]);

  useEffect(() => {
    if (view && !view.tags.some((t) => t.index === selected)) {
      setSelected(view.tags[0]?.index ?? 0);
    }
  }, [view, selected]);

  if (error) return <ErrorCard message={error} />;
  if (job?.status === "failed")
    return <ErrorCard message={job.error || "Обработка завершилась с ошибкой."} />;
  if (job?.status === "queued") return <Queued job={job} />;
  if (!job || job.status === "running") return <Processing job={job} />;
  if (!pred || !view) return <ReviewSkeleton />;

  return (
    <div className="flex flex-col gap-12">
      <Header job={job} count={view.tags.length} csvHref={jobsApi.csvUrl(id)} />
      <Summary data={view} />
      <Reviewer id={id} pred={view} selected={selected} onSelect={setSelected} />

      <Card>
        <CardHeader>
          <CardTitle>Все ценники</CardTitle>
        </CardHeader>
        <CardContent>
          <TagsTable data={view} selected={selected} onSelect={setSelected} />
        </CardContent>
      </Card>
    </div>
  );
}

// ===========================================================================
// Reviewer — one navigation model: the video's own timeline. Markers on the
// track are the tags; the playhead/selection is always derived from the
// frame on screen, so the overlay box can never go stale.
// ===========================================================================

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
  const pos = Math.max(0, order.findIndex((t) => t.index === selected));
  const tag = order[pos] ?? order[0];

  // "One frame" = tags whose best-frame falls in the same short time bucket.
  // The aggregator's per-tag best-frame timestamps cluster but rarely match
  // to the ms, so an exact-equality grouping would wrongly show every shelf
  // tag as a singleton. A tolerance bucket means a busy shelf frame is one
  // group with however MANY boxes it really has — the overlay + the in-frame
  // switcher below are written to scale to any count, not just 2–3.
  const frameKey = useCallback(
    (t: TagPrediction) => Math.round(t.frame_timestamp / FRAME_BUCKET_MS),
    [],
  );
  const framesByKey = useMemo(() => {
    const m = new Map<number, TagPrediction[]>();
    for (const t of order) {
      const k = Math.round(t.frame_timestamp / FRAME_BUCKET_MS);
      const arr = m.get(k) ?? [];
      arr.push(t);
      m.set(k, arr);
    }
    for (const arr of m.values()) arr.sort((a, b) => a.bbox.x1 - b.bbox.x1);
    return m;
  }, [order]);
  const mates = framesByKey.get(frameKey(tag)) ?? [tag];
  const boxPos = Math.max(0, mates.findIndex((t) => t.index === tag.index));

  const goTag = useCallback(
    (next: number) =>
      onSelect(order[Math.min(order.length - 1, Math.max(0, next))].index),
    [order, onSelect],
  );

  // ←/→ step between tags (never scrolls the page — no scrollIntoView/focus).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null;
      if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA")) return;
      if (e.key === "ArrowLeft") {
        e.preventDefault();
        goTag(pos - 1);
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        goTag(pos + 1);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [goTag, pos]);

  // Best-frame inspector by default; the user can switch to annotated
  // playback. The mode lives here so the video stage and the toggle stay
  // in lock-step.
  const [mode, setMode] = useState<ViewMode>("frame");

  return (
    <Card feature className="overflow-hidden">
      {/* Header — product name + which ценник of how many. No internal
          metrics: the timeline under the video is the one scrubber. */}
      <div className="flex flex-wrap items-end justify-between gap-4 border-b border-stone-border p-6">
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
            onClick={() => goTag(pos - 1)}
            disabled={pos === 0}
            title="Предыдущий ценник (←)"
          >
            <ChevronLeft /> Назад
          </Button>
          <span className="min-w-[4.5rem] text-center font-display text-heading-sm tabular-nums text-slate-text">
            {pos + 1} <span className="text-ash-gray">/ {order.length}</span>
          </span>
          <Button
            variant="outline"
            size="sm"
            onClick={() => goTag(pos + 1)}
            disabled={pos === order.length - 1}
            title="Следующий ценник (→)"
          >
            Вперёд <ChevronRight />
          </Button>
        </div>
      </div>

      <ViewModeBar mode={mode} onMode={setMode} />

      <div className="grid items-start gap-6 p-6 lg:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)]">
        <VideoStage
          id={id}
          pred={pred}
          tag={tag}
          mates={mates}
          frameKey={frameKey}
          mode={mode}
          onSelect={onSelect}
        />
        <DataPanel tag={tag} mates={mates} boxPos={boxPos} onSelect={onSelect} />
      </div>
    </Card>
  );
}

// --- View setting: how the video is shown ---------------------------------
//
// Two honest modes. "Лучший кадр" is the inspector — park on the chosen
// tag's best frame, its box spotlit. "Видео с разметкой" is annotated
// playback — the clip runs and each tag's box lights up at the moment it
// was read. Only above-threshold tags ever reach this screen (JobPage
// `view`), so in both modes only their boxes are drawn.

function ViewModeBar({
  mode,
  onMode,
}: {
  mode: ViewMode;
  onMode: (m: ViewMode) => void;
}) {
  const items: { id: ViewMode; label: string; icon: ReactNode }[] = [
    { id: "frame", label: "Лучший кадр", icon: <ImageIcon className="size-3.5" /> },
    { id: "live", label: "Видео с разметкой", icon: <Film className="size-3.5" /> },
  ];
  return (
    <div className="flex flex-wrap items-center justify-between gap-x-6 gap-y-3 border-b border-stone-border px-6 py-3">
      <div
        role="tablist"
        aria-label="Режим просмотра"
        className="inline-flex rounded-pill border border-stone-border bg-canvas-fog p-1"
      >
        {items.map((it) => {
          const active = it.id === mode;
          return (
            <button
              key={it.id}
              type="button"
              role="tab"
              aria-selected={active}
              onClick={() => onMode(it.id)}
              className={cn(
                "inline-flex cursor-pointer items-center gap-2 rounded-pill px-3.5 py-1.5 text-[13px] font-medium transition-colors",
                active
                  ? "bg-cloud-white text-slate-text shadow-subtle"
                  : "text-ash-gray hover:text-slate-text",
              )}
            >
              {it.icon}
              {it.label}
            </button>
          );
        })}
      </div>
      <p className="min-w-0 max-w-prose text-caption text-steel-gray">
        {mode === "frame"
          ? "Видео встаёт на лучший кадр выбранного ценника: его рамка выделена, остальное затемнено."
          : "Видео идёт как есть — рамка каждого ценника появляется в кадре, где он распознан. Нажмите на рамку, чтобы открыть данные."}
      </p>
    </div>
  );
}

// --- Video + its timeline (the only time-navigation control) --------------

function VideoStage({
  id,
  pred,
  tag,
  mates,
  frameKey,
  mode,
  onSelect,
}: {
  id: string;
  pred: JobPredictions;
  tag: TagPrediction;
  mates: TagPrediction[];
  frameKey: (t: TagPrediction) => number;
  mode: ViewMode;
  onSelect: (index: number) => void;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const trackRef = useRef<HTMLDivElement>(null);
  const [ready, setReady] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [scrubbing, setScrubbing] = useState(false);
  const [cur, setCur] = useState(0);
  const [dur, setDur] = useState(0);
  const [cropFailed, setCropFailed] = useState(false);

  const order = pred.tags;

  const timeOf = useCallback(
    (t: TagPrediction) =>
      Number.isFinite(dur) && dur > 0
        ? t.t_frac * dur
        : t.frame_timestamp / 1000,
    [dur],
  );

  const nearestTag = useCallback(
    (frac: number) => {
      let best = order[0];
      let bd = Infinity;
      for (const t of order) {
        const d = Math.abs(t.t_frac - frac);
        if (d < bd) {
          bd = d;
          best = t;
        }
      }
      return best;
    },
    [order],
  );

  const drawCrop = useCallback(() => {
    const v = videoRef.current;
    const c = canvasRef.current;
    if (!v || !c) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, c.width, c.height); // never show the previous tag
    if (!v.videoWidth || !v.videoHeight) return;
    const { x1, y1, x2, y2 } = tag.bbox;
    const sx = x1 * v.videoWidth;
    const sy = y1 * v.videoHeight;
    const sw = Math.max(1, (x2 - x1) * v.videoWidth);
    const sh = Math.max(1, (y2 - y1) * v.videoHeight);
    const W = 520;
    c.width = W;
    c.height = Math.max(1, Math.round((W * sh) / sw));
    try {
      ctx.drawImage(v, sx, sy, sw, sh, 0, 0, c.width, c.height);
      setCropFailed(false);
    } catch {
      setCropFailed(true); // cross-origin taint — only in a broken deploy
    }
  }, [tag.bbox]);

  // Park the clip on the selected tag's frame (an explicit pick always
  // pauses + seeks, so the frame and the data panel stay in lock-step).
  // ONLY in best-frame mode: in live mode the clip must keep playing, so a
  // pick / box-click must never yank the playhead. `mode` is a dep so that
  // switching back to best-frame re-parks on the open tag.
  useEffect(() => {
    if (mode !== "frame") return;
    const v = videoRef.current;
    if (!v || !ready) return;
    v.pause();
    const t = timeOf(tag);
    v.currentTime = Math.min(
      Number.isFinite(dur) && dur > 0 ? dur - 0.04 : t,
      Math.max(0, t),
    );
  }, [tag, ready, dur, timeOf, mode]);

  // The frame on screen is the source of truth in best-frame mode: after any
  // seek/pause, snap the selection to the tag at that moment — unless we're
  // still on the same frame (then keep the chosen box among its mates). This
  // is what makes a stale overlay impossible. In live mode the playhead is
  // free: we only keep `cur` (→ playhead + live boxes) current and never
  // reselect or redraw the crop, so playback is never fought.
  const syncToFrame = useCallback(() => {
    const v = videoRef.current;
    if (!v || !dur) return;
    setCur(v.currentTime);
    if (mode !== "frame") return;
    const near = nearestTag(v.currentTime / dur);
    // Only re-select when we've moved to a different frame; staying on the
    // same busy frame keeps whichever of its many boxes the user picked.
    if (frameKey(near) !== frameKey(tag)) onSelect(near.index);
    drawCrop();
  }, [dur, mode, nearestTag, onSelect, frameKey, tag, drawCrop]);

  const seekToFrac = useCallback(
    (frac: number) => {
      const v = videoRef.current;
      if (!v || !dur) return;
      v.currentTime = Math.min(dur - 0.04, Math.max(0, frac * dur));
    },
    [dur],
  );

  const onTrackPointer = useCallback(
    (e: ReactPointerEvent) => {
      const el = trackRef.current;
      if (!el) return;
      const r = el.getBoundingClientRect();
      seekToFrac((e.clientX - r.left) / r.width);
    },
    [seekToFrac],
  );

  const togglePlay = useCallback(() => {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) void v.play();
    else v.pause();
  }, []);

  // Best-frame mode is an inspector: boxes only when the clip is parked.
  // Live mode IS the annotated playback: boxes stay on while playing AND
  // while scrubbing (that's the whole point of the mode).
  const boxesVisible =
    mode === "live" ? ready : ready && !playing && !scrubbing;

  // The tags whose captured moment is within the display window of the
  // playhead. `order` is already threshold-filtered upstream (JobPage
  // `view`), so this shows ONLY boxes that passed the metric threshold.
  const liveBoxes = useMemo(() => {
    if (mode !== "live") return [] as TagPrediction[];
    return order.filter((t) => Math.abs(cur - timeOf(t)) <= LIVE_WINDOW_S);
  }, [mode, order, cur, timeOf]);

  return (
    <div className="flex flex-col gap-3">
      {/* The wrapper shrink-wraps the <video> (w-fit) so the % overlay maps
          to the real video pixels — no object-contain letterbox drift even
          for a tall portrait clip with many boxes. */}
      <div className="mx-auto flex w-full justify-center">
        <div className="relative w-fit overflow-hidden rounded-input bg-ghost-ink">
        <video
          ref={videoRef}
          src={jobsApi.videoUrl(id)}
          crossOrigin="anonymous"
          playsInline
          preload="auto"
          className="block max-h-[60vh] max-w-full"
          onLoadedMetadata={(e) => {
            setDur(e.currentTarget.duration || 0);
            setReady(true);
          }}
          onLoadedData={drawCrop}
          onTimeUpdate={(e) => setCur(e.currentTarget.currentTime)}
          onSeeked={(e) => {
            // Keep the playhead + live boxes tracking even during a paused
            // scrub (timeupdate may not fire then); only frame-snap once the
            // scrub is released.
            setCur(e.currentTarget.currentTime);
            if (!scrubbing) syncToFrame();
          }}
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onClick={togglePlay}
        />

        {/* Boxes for THIS frame only. Active one is focused via a clipped
            9999px shadow (everything else dims); its mates are thin
            outlines you can click to switch boxes within the frame. */}
        {boxesVisible &&
          (mode === "frame" ? mates : liveBoxes).map((m) => {
            const active = m.index === tag.index;
            // Spotlight (dim everything else) is a best-frame inspector
            // affordance — never dim the shelf while the clip is playing.
            const spotlight = mode === "frame" && active;
            return (
              <button
                key={m.index}
                type="button"
                title={tagLabel(m)}
                onClick={(e) => {
                  e.stopPropagation();
                  onSelect(m.index);
                }}
                className={cn(
                  "absolute rounded-[3px] transition-colors",
                  active
                    ? "border-2 border-chartwell-blue"
                    : "cursor-pointer border border-cloud-white/70 hover:border-2 hover:border-chartwell-blue",
                )}
                style={{
                  left: `${m.bbox.x1 * 100}%`,
                  top: `${m.bbox.y1 * 100}%`,
                  width: `${(m.bbox.x2 - m.bbox.x1) * 100}%`,
                  height: `${(m.bbox.y2 - m.bbox.y1) * 100}%`,
                  boxShadow: spotlight
                    ? "0 0 0 9999px rgba(12, 10, 9, 0.55)"
                    : undefined,
                }}
              />
            );
          })}

        {playing && mode === "frame" && (
          <div className="pointer-events-none absolute left-3 top-3 rounded-pill bg-ghost-ink/70 px-2.5 py-1 text-[12px] font-medium text-cloud-white">
            Воспроизведение — рамки скрыты
          </div>
        )}
        {mode === "live" && (
          <div className="pointer-events-none absolute left-3 top-3 inline-flex items-center gap-1.5 rounded-pill bg-ghost-ink/70 px-2.5 py-1 text-[12px] font-medium text-cloud-white">
            <span className="size-1.5 rounded-full bg-chartwell-blue" />
            Разметка по времени
          </div>
        )}
        </div>
      </div>

      {/* The timeline IS the scrubber: each dot is a ценник, the blue one is
          open. Click a dot to open it; drag the bar to move through video. */}
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={togglePlay}
          title={playing ? "Пауза" : "Воспроизвести"}
          className="grid size-9 shrink-0 cursor-pointer place-items-center rounded-pill border border-stone-border bg-cloud-white text-slate-text transition-colors hover:border-chartwell-blue hover:text-chartwell-blue"
        >
          {playing ? (
            <Pause className="size-4" />
          ) : (
            <Play className="size-4" />
          )}
        </button>
        <div
          ref={trackRef}
          onPointerDown={(e) => {
            (e.target as HTMLElement).setPointerCapture(e.pointerId);
            setScrubbing(true);
            onTrackPointer(e);
          }}
          onPointerMove={(e) => scrubbing && onTrackPointer(e)}
          onPointerUp={(e) => {
            (e.target as HTMLElement).releasePointerCapture(e.pointerId);
            setScrubbing(false);
            syncToFrame();
          }}
          className="relative h-9 grow cursor-pointer select-none"
        >
          {/* rail */}
          <div className="absolute inset-x-0 top-1/2 h-1.5 -translate-y-1/2 rounded-pill bg-stone-border" />
          {/* played */}
          <div
            className="absolute left-0 top-1/2 h-1.5 -translate-y-1/2 rounded-pill bg-chartwell-blue/70"
            style={{ width: `${dur ? (cur / dur) * 100 : 0}%` }}
          />
          {/* one dot per ценник; the open one is blue, the rest are
              neutral and grow on hover so they read as clickable */}
          {order.map((t) => {
            const active = t.index === tag.index;
            return (
              <button
                key={t.index}
                type="button"
                title={`${tagLabel(t)} — лучший кадр`}
                onPointerDown={(e) => e.stopPropagation()}
                onClick={(e) => {
                  e.stopPropagation();
                  onSelect(t.index);
                  // In live mode opening a tag doesn't park the clip, so jump
                  // the playhead to its moment — "go here and watch it".
                  if (mode === "live") seekToFrac(t.t_frac);
                }}
                className={cn(
                  "absolute top-1/2 -translate-x-1/2 -translate-y-1/2 cursor-pointer rounded-full border-2 border-cloud-white transition-all",
                  active
                    ? "z-10 size-4 bg-chartwell-blue"
                    : "size-2.5 bg-steel-gray hover:size-3.5 hover:bg-slate-text",
                )}
                style={{ left: `${t.t_frac * 100}%` }}
              />
            );
          })}
          {/* playhead */}
          <div
            className="pointer-events-none absolute top-1/2 size-4 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-chartwell-blue bg-cloud-white shadow-subtle"
            style={{ left: `${dur ? (cur / dur) * 100 : 0}%` }}
          />
        </div>
        <span className="shrink-0 font-mono text-[12px] tabular-nums text-ash-gray">
          {clock(cur)} / {clock(dur)}
        </span>
      </div>
      <p className="text-caption text-steel-gray">
        Каждая точка — лучший кадр ценника: именно по нему распознаны данные
        и взято время в видео. Нажмите, чтобы открыть; тяните дорожку, чтобы
        перемотать.
      </p>

      {/* The cropped tag image — a best-frame inspector affordance. In live
          mode it would just flicker through frames, so it's not rendered
          (the canvas ref is null then; drawCrop is guarded). */}
      {mode === "frame" && (
        <div className="mt-2">
          <p className="mb-2 flex items-center gap-2 text-caption text-ash-gray">
            <Crop className="size-3.5" /> Изображение ценника
          </p>
          <div className="inline-block overflow-hidden rounded-input border border-stone-border bg-canvas-fog">
            <canvas ref={canvasRef} className="block max-h-[220px] max-w-full" />
          </div>
          {cropFailed && (
            <p className="mt-2 text-caption text-amber-700">
              Не удалось показать изображение ценника.
            </p>
          )}
        </div>
      )}
    </div>
  );
}

// --- Recognized data — hero facts + folded detail groups ------------------

function DataPanel({
  tag,
  mates,
  boxPos,
  onSelect,
}: {
  tag: TagPrediction;
  mates: TagPrediction[];
  boxPos: number;
  onSelect: (index: number) => void;
}) {
  const cm = colorMeta(tag.color);
  return (
    <div className="flex flex-col gap-5">
      {/* Boxes-in-one-frame switcher (only when this frame holds several). */}
      {mates.length > 1 && (
        <div className="flex items-center justify-between rounded-input border border-stone-border bg-canvas-fog px-3 py-2">
          <span className="text-caption text-ash-gray">
            В этом кадре несколько ценников
          </span>
          <div className="flex items-center gap-2">
            <button
              type="button"
              title="Предыдущий ценник в кадре"
              onClick={() =>
                onSelect(mates[(boxPos - 1 + mates.length) % mates.length].index)
              }
              className="grid size-7 cursor-pointer place-items-center rounded-pill border border-stone-border bg-cloud-white text-slate-text transition-colors hover:border-chartwell-blue hover:text-chartwell-blue"
            >
              <ChevronLeft className="size-3.5" />
            </button>
            <span className="text-[12px] tabular-nums text-slate-text">
              {boxPos + 1} из {mates.length}
            </span>
            <button
              type="button"
              title="Следующий ценник в кадре"
              onClick={() => onSelect(mates[(boxPos + 1) % mates.length].index)}
              className="grid size-7 cursor-pointer place-items-center rounded-pill border border-stone-border bg-cloud-white text-slate-text transition-colors hover:border-chartwell-blue hover:text-chartwell-blue"
            >
              <ChevronRight className="size-3.5" />
            </button>
          </div>
        </div>
      )}

      {/* Hero — the few things that matter, read at a glance. */}
      <div className="rounded-card border border-stone-border bg-canvas-fog p-4">
        {tag.color.toLowerCase() !== "white" && (
          <Badge variant="accent" className="mb-3">
            {cm.label}
          </Badge>
        )}
        <div className="grid grid-cols-2 gap-x-4 gap-y-4">
          <HeroStat label="Цена без карты" value={tag.fields.price_default} suffix="₽" />
          <HeroStat label="Цена по карте" value={tag.fields.price_card} suffix="₽" />
          <HeroStat label="Штрихкод" value={tag.fields.barcode} mono />
          <HeroStat label="Артикул (SKU)" value={tag.fields.id_sku} mono />
        </div>
      </div>

      {/* Detail groups — empty groups collapse so the panel never becomes a
          giant column (QR-код is almost always "нет данных"). */}
      <div className="flex flex-col gap-3">
        {FIELD_GROUPS.map((group) => {
          const rows = group.fields.filter(
            (f) => !HIDDEN_FROM_GROUPS.has(f.key),
          );
          if (!rows.length) return null;
          const filled = rows.filter(
            (f) => fieldState(tag.fields[f.key]) === "value",
          ).length;
          return (
            <details
              key={group.title}
              open={filled > 0}
              className="group overflow-hidden rounded-input border border-stone-border"
            >
              <summary className="flex cursor-pointer list-none items-center justify-between bg-cloud-white px-3 py-2 text-caption font-medium uppercase tracking-[0.12em] text-steel-gray">
                <span>{group.title}</span>
                <span className="flex items-center gap-2 normal-case tracking-normal">
                  <span className="text-[12px] text-ash-gray">
                    {filled > 0 ? `${filled} значений` : "нет данных"}
                  </span>
                  <ChevronRight className="size-3.5 text-steel-gray transition-transform group-open:rotate-90" />
                </span>
              </summary>
              <dl className="grid grid-cols-1 gap-px bg-stone-border sm:grid-cols-2">
                {rows.map(({ key, label }) => (
                  <div
                    key={key}
                    className="flex items-baseline justify-between gap-3 bg-cloud-white px-3 py-2"
                  >
                    <dt className="shrink-0 text-[13px] text-ash-gray">
                      {label}
                    </dt>
                    <dd className="min-w-0 text-right">
                      <FieldValue field={key} value={tag.fields[key]} />
                    </dd>
                  </div>
                ))}
              </dl>
            </details>
          );
        })}
      </div>
    </div>
  );
}

function HeroStat({
  label,
  value,
  suffix,
  mono,
}: {
  label: string;
  value: string | undefined;
  suffix?: string;
  mono?: boolean;
}) {
  const state = fieldState(value);
  return (
    <div className="min-w-0">
      <p className="text-caption text-ash-gray">{label}</p>
      {state === "value" ? (
        <p
          className={cn(
            "mt-1 truncate font-display text-heading-sm font-medium text-slate-text",
            mono && "font-mono tabular-nums",
          )}
        >
          {value}
          {suffix && <span className="ml-1 text-[13px] text-ash-gray">{suffix}</span>}
        </p>
      ) : (
        <p className="mt-1 text-[13px] text-steel-gray">
          {state === "absent" ? "нет на ценнике" : "не распознано"}
        </p>
      )}
    </div>
  );
}

function FieldValue({ field, value }: { field: string; value: string | undefined }) {
  const state = fieldState(value);
  if (state === "absent")
    return (
      <Badge variant="neutral" className="font-normal">
        нет на ценнике
      </Badge>
    );
  if (state === "unrecognized")
    return (
      <Badge variant="warning" className="font-normal">
        не распознано
      </Badge>
    );
  const mono =
    field === "barcode" || field.includes("qr") || field === "id_sku";
  return (
    <span
      className={cn(
        "break-words text-[13px] text-slate-text",
        mono && "font-mono tabular-nums",
      )}
    >
      {displayValue(field, value ?? "")}
    </span>
  );
}

function clock(sec: number): string {
  if (!Number.isFinite(sec) || sec < 0) return "0:00";
  const s = Math.floor(sec % 60);
  return `${Math.floor(sec / 60)}:${s.toString().padStart(2, "0")}`;
}

// ===========================================================================

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
          Скачать таблицу (CSV)
        </Button>
      </a>
    </div>
  );
}

// Real ML stages → honest RU labels. The end-of-video Qwen burst
// ("finalize") used to hide behind a frozen "Сборка выгрузки"; it is now
// its own moving stage. Falls back to fraction guesses only when the
// backend gives no phase (MOCK_MODE / before the first poll).
const PHASE_LABEL: Record<string, string> = {
  detect: "Ищем и ведём ценники в кадрах…",
  finalize: "Распознаём ценники нейросетью…",
  dedup: "Склеиваем повторы одного ценника…",
  done: "Формируем результат…",
};

// Waiting its turn — the single worker is busy with an earlier video. This
// is a deliberate product limit (one cheap GPU), so it is explained, not
// hidden behind a frozen progress bar.
function Queued({ job }: { job: Job }) {
  const pos = job.queue_position;
  const line =
    pos == null
      ? "Готовим видео к обработке…"
      : pos <= 0
        ? "Вы следующий — обработка начнётся, как только освободится сервер."
        : "Дождёмся их обработки и сразу возьмёмся за ваше видео.";
  return (
    <div className="grid place-items-center py-24">
      <Card feature className="w-full max-w-md p-8 text-center">
        <div className="mx-auto grid size-12 place-items-center rounded-card bg-chartwell-blue/10 text-chartwell-blue">
          <Clock className="size-6" />
        </div>
        <h1 className="mt-5 font-display text-heading font-medium text-slate-text">
          Видео в очереди
        </h1>
        <p className="mt-1 truncate text-caption text-ash-gray">
          {job.filename}
        </p>
        {pos != null && pos > 0 && (
          <p className="mt-5 font-display text-display font-medium tabular-nums text-slate-text">
            {pos}
            <span className="ml-2 align-middle text-caption font-normal text-ash-gray">
              {pos === 1 ? "видео впереди" : "видео в очереди перед вами"}
            </span>
          </p>
        )}
        <p className="mt-4 text-[14px] leading-[1.6] text-slate-text">{line}</p>
        <ServerLimitNote className="mt-6 text-left" />
        <p className="mt-4 text-caption text-steel-gray">
          Экран обновится сам — страницу можно не перезагружать.
        </p>
      </Card>
    </div>
  );
}

function Processing({ job }: { job: Job | null }) {
  const progress = job?.progress ?? 0;
  const phase = !job
    ? "Загружаем статус…"
    : (job.phase && PHASE_LABEL[job.phase]) ||
      (progress < 0.5
        ? "Ищем ценники в кадрах…"
        : progress < 0.9
          ? "Распознаём поля и штрихкоды…"
          : "Завершаем…");
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
    <div className="flex flex-col gap-12">
      <Skeleton className="h-12 w-72" />
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <Skeleton key={i} className="h-28" />
        ))}
      </div>
      <Skeleton className="h-[560px]" />
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
