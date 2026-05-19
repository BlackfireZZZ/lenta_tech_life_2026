import {
  type PointerEvent as ReactPointerEvent,
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
  Crop,
  Download,
  Image as ImageIcon,
  Pause,
  Play,
  ScanLine,
} from "lucide-react";
import {
  type DetectorBox,
  type DetectorTrace,
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
import { JOB_PAGE_COPY } from "@/content/jobPageContent";

const POLL_MS = 1200;

// Tags whose best-frame timestamps fall within this window are treated as
// "the same frame" (so a busy shelf moment is one group of many boxes).
const FRAME_BUCKET_MS = 200;

// Two honest review modes (a segmented switch by the video):
//  - "frame"    — the existing inspector: one best frame per ценник.
//  - "detector" — play the clip with the raw per-frame detector output
//    (every box above its confidence threshold) drawn live, to judge the
//    detector on its own. Its data is a separate non-graded trace fetched
//    from the backend; the graded CSV is never derived from it.
type ViewMode = "frame" | "detector";
type TraceState = "idle" | "loading" | "ready" | "error";

// In the detector view a trace frame is matched to the playhead by nearest
// timestamp. Frames with no detection are omitted from the trace, so "no
// frame within tolerance" honestly means the detector found nothing there —
// we then draw nothing (never a stale box). Widened for a strided long-clip
// trace from the real per-frame gap.
const DET_BASE_TOLERANCE_MS = 500;

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
        if (alive) setError(JOB_PAGE_COPY.errors.notFound);
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
      .catch(() => alive && setError(JOB_PAGE_COPY.errors.predictions));
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
    return <ErrorCard message={job.error || JOB_PAGE_COPY.errors.failedDefault} />;
  if (!job || job.status === "queued" || job.status === "running")
    return <Processing job={job} />;
  if (!pred || !view) return <ReviewSkeleton />;

  return (
    <div className="flex flex-col gap-12">
      <Header job={job} count={view.tags.length} csvHref={jobsApi.csvUrl(id)} />
      <Summary data={view} />
      <Reviewer id={id} pred={view} selected={selected} onSelect={setSelected} />

      <Card>
        <CardHeader>
          <CardTitle>{JOB_PAGE_COPY.tableTitle}</CardTitle>
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

  // View mode + the detector trace. The trace can be sizeable, so it is
  // fetched lazily — only the first time the user opens the detector view —
  // and never re-fetched (it is immutable for a finished job). A 409 / any
  // error just disables the detector view; the best-frame inspector and the
  // graded CSV are wholly independent and unaffected.
  const [mode, setMode] = useState<ViewMode>("frame");
  const [trace, setTrace] = useState<DetectorTrace | null>(null);
  const [traceState, setTraceState] = useState<TraceState>("idle");

  useEffect(() => {
    if (mode !== "detector" || trace || traceState !== "idle") return;
    let alive = true;
    setTraceState("loading");
    jobsApi
      .getDetections(id)
      .then((t) => {
        if (!alive) return;
        setTrace(t);
        setTraceState("ready");
      })
      .catch(() => alive && setTraceState("error"));
    return () => {
      alive = false;
    };
  }, [mode, trace, traceState, id]);

  return (
    <Card feature className="overflow-hidden">
      {/* Header — product name + which ценник of how many. No internal
          metrics: the timeline under the video is the one scrubber. */}
      <div className="flex flex-wrap items-end justify-between gap-4 border-b border-stone-border p-6">
        <div className="min-w-0">
          <p className="text-caption uppercase tracking-[0.12em] text-steel-gray">
            {JOB_PAGE_COPY.reviewer.title}
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
            title={JOB_PAGE_COPY.reviewer.prevTagTitle}
          >
            <ChevronLeft /> {JOB_PAGE_COPY.reviewer.prevTagButton}
          </Button>
          <span className="min-w-[4.5rem] text-center font-display text-heading-sm tabular-nums text-slate-text">
            {pos + 1} <span className="text-ash-gray">/ {order.length}</span>
          </span>
          <Button
            variant="outline"
            size="sm"
            onClick={() => goTag(pos + 1)}
            disabled={pos === order.length - 1}
            title={JOB_PAGE_COPY.reviewer.nextTagTitle}
          >
            {JOB_PAGE_COPY.reviewer.nextTagButton} <ChevronRight />
          </Button>
        </div>
      </div>

      <ViewModeBar mode={mode} onMode={setMode} sampled={trace?.sampled} />

      <div className="grid items-start gap-6 px-6 pb-6 lg:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)]">
        <VideoStage
          id={id}
          pred={pred}
          tag={tag}
          mates={mates}
          frameKey={frameKey}
          onSelect={onSelect}
          mode={mode}
          trace={trace}
          traceState={traceState}
        />
        <DataPanel tag={tag} mates={mates} boxPos={boxPos} onSelect={onSelect} />
      </div>
    </Card>
  );
}

// --- The two-mode switch (DESIGN.md: subtle ash-tinted segmented control,
// pill radius; the active segment lifts to Cloud White; Chartwell Blue only
// on the active label/icon). Sits on its own row under the header so it
// reads as "how you look at this", not a tag control. ----------------------

function ViewModeBar({
  mode,
  onMode,
  sampled,
}: {
  mode: ViewMode;
  onMode: (m: ViewMode) => void;
  sampled?: boolean;
}) {
  const C = JOB_PAGE_COPY.viewMode;
  const items: { key: ViewMode; label: string; icon: typeof ImageIcon }[] = [
    { key: "frame", label: C.bestFrame, icon: ImageIcon },
    { key: "detector", label: C.detector, icon: ScanLine },
  ];
  return (
    <div className="flex flex-col gap-2 px-6 pt-6 pb-4">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <span className="text-caption uppercase tracking-[0.12em] text-steel-gray">
          {C.label}
        </span>
        <div
          role="tablist"
          aria-label={C.label}
          className="inline-flex gap-1 rounded-pill border border-stone-border bg-ash-gray/10 p-1"
        >
          {items.map(({ key, label, icon: Icon }) => {
            const active = mode === key;
            return (
              <button
                key={key}
                type="button"
                role="tab"
                aria-selected={active}
                onClick={() => onMode(key)}
                className={cn(
                  "inline-flex cursor-pointer items-center gap-1.5 rounded-pill px-3 py-1.5 text-[13px] font-medium transition-colors",
                  active
                    ? "bg-cloud-white text-chartwell-blue shadow-subtle"
                    : "text-ash-gray hover:text-slate-text",
                )}
              >
                <Icon className="size-3.5" />
                {label}
              </button>
            );
          })}
        </div>
      </div>
      <p className="text-caption text-steel-gray">
        {mode === "detector" ? C.detectorHint : C.bestFrameHint}
        {mode === "detector" && sampled ? ` · ${C.sampledNote}` : ""}
      </p>
    </div>
  );
}

// RU plural for the live "N рамок" pill (1 рамка / 2–4 рамки / 5+ рамок).
function boxesWord(n: number): string {
  const C = JOB_PAGE_COPY.viewMode;
  const m100 = n % 100;
  const m10 = n % 10;
  if (m100 >= 11 && m100 <= 14) return C.boxesMany;
  if (m10 === 1) return C.boxesOne;
  if (m10 >= 2 && m10 <= 4) return C.boxesFew;
  return C.boxesMany;
}

// --- Video + its timeline (the only time-navigation control) --------------

function VideoStage({
  id,
  pred,
  tag,
  mates,
  frameKey,
  onSelect,
  mode,
  trace,
  traceState,
}: {
  id: string;
  pred: JobPredictions;
  tag: TagPrediction;
  mates: TagPrediction[];
  frameKey: (t: TagPrediction) => number;
  onSelect: (index: number) => void;
  mode: ViewMode;
  trace: DetectorTrace | null;
  traceState: TraceState;
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
  // Best-frame mode only: in the detector view the clip is meant to play
  // freely, so selecting a tag must never yank playback. Switching back to
  // "frame" re-parks (mode is in the deps), restoring the inspector exactly.
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

  // The frame on screen is the source of truth: after any seek/pause, snap
  // the selection to the tag at that moment — unless we're still on the same
  // frame (then keep the chosen box among its mates). This is what makes a
  // stale overlay impossible.
  const syncToFrame = useCallback(() => {
    const v = videoRef.current;
    if (!v || !dur) return;
    setCur(v.currentTime); // keep the playhead/clock honest in both modes
    if (mode !== "frame") return; // detector view derives its overlay from cur
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

  // Detector trace, indexed for a fast nearest-by-time lookup. The match
  // tolerance widens to the trace's own median frame gap (a strided long
  // clip has larger gaps) so boxes don't flicker between sampled frames.
  const detTimes = useMemo(
    () => (trace ? trace.frames.map((f) => f.t_ms) : []),
    [trace],
  );
  const detTolMs = useMemo(() => {
    if (!trace || trace.frames.length < 2) return DET_BASE_TOLERANCE_MS;
    const gaps: number[] = [];
    for (let i = 1; i < detTimes.length; i++)
      gaps.push(detTimes[i] - detTimes[i - 1]);
    gaps.sort((a, b) => a - b);
    const med = gaps[gaps.length >> 1] || DET_BASE_TOLERANCE_MS;
    return Math.min(2500, Math.max(DET_BASE_TOLERANCE_MS, Math.round(med * 1.5)));
  }, [trace, detTimes]);

  // Boxes for the frame nearest the playhead (binary search over the
  // time-ordered trace). Empty when no frame is within tolerance — that
  // honestly means the detector found nothing there; we draw nothing.
  const detBoxes = useMemo<DetectorBox[]>(() => {
    if (mode !== "detector" || !trace || detTimes.length === 0) return [];
    const ms = cur * 1000;
    let lo = 0;
    let hi = detTimes.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (detTimes[mid] < ms) lo = mid + 1;
      else hi = mid;
    }
    let best = lo;
    if (lo > 0 && Math.abs(detTimes[lo - 1] - ms) <= Math.abs(detTimes[lo] - ms))
      best = lo - 1;
    return Math.abs(detTimes[best] - ms) <= detTolMs
      ? trace.frames[best].boxes
      : [];
  }, [mode, trace, detTimes, detTolMs, cur]);

  // While the detector clip plays, drive the overlay from rAF (throttled to
  // ~16 fps) — `timeupdate` fires only ~4 Hz, which makes boxes lurch. Off
  // in best-frame mode (boxes are hidden during playback there anyway).
  useEffect(() => {
    if (mode !== "detector" || !playing) return;
    const v = videoRef.current;
    if (!v) return;
    let raf = 0;
    let last = -1;
    const tick = () => {
      const t = v.currentTime;
      if (Math.abs(t - last) >= 0.05) {
        last = t;
        setCur(t);
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [mode, playing]);

  // Frame mode keeps the original rule (boxes only on a settled frame).
  // Detector mode is the opposite: the boxes ARE the point during playback.
  const boxesVisible = ready && !playing && !scrubbing;
  const detReady = mode === "detector" && ready;

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
            // Always advance the playhead so the detector overlay tracks a
            // seek/scrub too; the frame-mode re-select stays gated in
            // syncToFrame (and is skipped mid-scrub).
            setCur(e.currentTarget.currentTime);
            if (!scrubbing) syncToFrame();
          }}
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onClick={togglePlay}
        />

        {/* FRAME MODE — boxes for THIS frame only. Active one is focused via
            a clipped 9999px shadow (everything else dims); its mates are
            thin outlines you can click to switch boxes within the frame.
            Unchanged from the original inspector. */}
        {mode === "frame" &&
          boxesVisible &&
          mates.map((m) => {
            const active = m.index === tag.index;
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
                  boxShadow: active
                    ? "0 0 0 9999px rgba(12, 10, 9, 0.55)"
                    : undefined,
                }}
              />
            );
          })}

        {/* DETECTOR MODE — every above-threshold detector box for the frame
            at the playhead. Non-interactive (pointer-events-none) so a click
            still toggles play; this view is for watching, not picking. */}
        {detReady &&
          detBoxes.map((b, i) => {
            const [x1, y1, x2, y2, score] = b;
            const wide = x2 - x1 >= 0.06 && y2 - y1 >= 0.03;
            return (
              <div
                key={i}
                className="pointer-events-none absolute rounded-[3px] border border-chartwell-blue bg-chartwell-blue/5"
                style={{
                  left: `${x1 * 100}%`,
                  top: `${y1 * 100}%`,
                  width: `${(x2 - x1) * 100}%`,
                  height: `${(y2 - y1) * 100}%`,
                }}
              >
                {wide && (
                  <span className="absolute -top-[18px] left-0 rounded-[3px] bg-chartwell-blue px-1 text-[10px] font-medium leading-[16px] text-primary-foreground tabular-nums">
                    {Math.round(score * 100)}%
                  </span>
                )}
              </div>
            );
          })}

        {/* FRAME MODE — playback hides the boxes (the original behaviour). */}
        {mode === "frame" && playing && (
          <div className="pointer-events-none absolute left-3 top-3 rounded-pill bg-ghost-ink/70 px-2.5 py-1 text-[12px] font-medium text-cloud-white">
            {JOB_PAGE_COPY.reviewer.playingOverlay}
          </div>
        )}

        {/* DETECTOR MODE — live status pill (count + the honest threshold),
            or a loading / unavailable note. */}
        {mode === "detector" && traceState === "loading" && (
          <div className="pointer-events-none absolute inset-0 grid place-items-center bg-ghost-ink/40">
            <span className="inline-flex items-center gap-2 rounded-pill bg-ghost-ink/70 px-3 py-1.5 text-[12px] font-medium text-cloud-white">
              <Spinner className="size-3.5" />
              {JOB_PAGE_COPY.viewMode.loading}
            </span>
          </div>
        )}
        {mode === "detector" &&
          (traceState === "error" ||
            (traceState === "ready" && trace?.frames.length === 0)) && (
            <div className="pointer-events-none absolute inset-x-3 top-3 rounded-pill bg-ghost-ink/70 px-3 py-1.5 text-center text-[12px] font-medium text-cloud-white">
              {JOB_PAGE_COPY.viewMode.unavailable}
            </div>
          )}
        {detReady && traceState === "ready" && (trace?.frames.length ?? 0) > 0 && (
          <div className="pointer-events-none absolute left-3 top-3 rounded-pill bg-ghost-ink/70 px-2.5 py-1 text-[12px] font-medium text-cloud-white tabular-nums">
            {detBoxes.length > 0
              ? `${detBoxes.length} ${boxesWord(detBoxes.length)} ${JOB_PAGE_COPY.viewMode.inFrame} · ${JOB_PAGE_COPY.viewMode.threshold} ${trace!.conf_threshold.toFixed(2)}`
              : playing
                ? JOB_PAGE_COPY.viewMode.noBoxesHere
                : JOB_PAGE_COPY.viewMode.play}
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
          title={playing ? JOB_PAGE_COPY.reviewer.pauseTitle : JOB_PAGE_COPY.reviewer.playTitle}
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
                title={`${tagLabel(t)} ${JOB_PAGE_COPY.reviewer.pointBestFrameSuffix}`}
                onPointerDown={(e) => e.stopPropagation()}
                onClick={(e) => {
                  e.stopPropagation();
                  onSelect(t.index);
                  // Frame mode: selecting parks the clip (gated park effect).
                  // Detector mode: nothing parks, so jump there explicitly —
                  // the dot stays a useful "go to this ценник" control
                  // without interrupting playback.
                  if (mode === "detector") seekToFrac(t.t_frac);
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
        {JOB_PAGE_COPY.reviewer.timelineHint}
      </p>

      {/* The cropped tag image — a best-frame artifact. Hidden in the
          detector view, where the clip sits at an arbitrary play position
          and a per-tag crop would be misleading. */}
      {mode === "frame" && (
        <div className="mt-2">
          <p className="mb-2 flex items-center gap-2 text-caption text-ash-gray">
            <Crop className="size-3.5" /> {JOB_PAGE_COPY.reviewer.cropTitle}
          </p>
          <div className="inline-block overflow-hidden rounded-input border border-stone-border bg-canvas-fog">
            <canvas ref={canvasRef} className="block max-h-[220px] max-w-full" />
          </div>
          {cropFailed && (
            <p className="mt-2 text-caption text-amber-700">
              {JOB_PAGE_COPY.reviewer.cropFailed}
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
            {JOB_PAGE_COPY.frameSwitcher.manyInFrame}
          </span>
          <div className="flex items-center gap-2">
            <button
              type="button"
              title={JOB_PAGE_COPY.frameSwitcher.prevInFrameTitle}
              onClick={() =>
                onSelect(mates[(boxPos - 1 + mates.length) % mates.length].index)
              }
              className="grid size-7 cursor-pointer place-items-center rounded-pill border border-stone-border bg-cloud-white text-slate-text transition-colors hover:border-chartwell-blue hover:text-chartwell-blue"
            >
              <ChevronLeft className="size-3.5" />
            </button>
            <span className="text-[12px] tabular-nums text-slate-text">
              {boxPos + 1} {JOB_PAGE_COPY.frameSwitcher.indexDivider} {mates.length}
            </span>
            <button
              type="button"
              title={JOB_PAGE_COPY.frameSwitcher.nextInFrameTitle}
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
          <HeroStat label={JOB_PAGE_COPY.heroStats.priceDefault} value={tag.fields.price_default} suffix="₽" />
          <HeroStat label={JOB_PAGE_COPY.heroStats.priceCard} value={tag.fields.price_card} suffix="₽" />
          <HeroStat label={JOB_PAGE_COPY.heroStats.barcode} value={tag.fields.barcode} mono />
          <HeroStat label={JOB_PAGE_COPY.heroStats.sku} value={tag.fields.id_sku} mono />
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
                    {filled > 0
                      ? `${filled} ${JOB_PAGE_COPY.groups.valuesSuffix}`
                      : JOB_PAGE_COPY.groups.noData}
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
          {state === "absent"
            ? JOB_PAGE_COPY.fieldState.absent
            : JOB_PAGE_COPY.fieldState.unrecognized}
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
        {JOB_PAGE_COPY.fieldState.absent}
      </Badge>
    );
  if (state === "unrecognized")
    return (
      <Badge variant="warning" className="font-normal">
        {JOB_PAGE_COPY.fieldState.unrecognized}
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
          <ArrowLeft className="size-3.5" /> {JOB_PAGE_COPY.header.newVideo}
        </Link>
        <h1 className="mt-2 truncate font-display text-heading-lg font-medium text-slate-text">
          {job.filename}
        </h1>
        <div className="mt-2 flex items-center gap-2">
          <Badge variant="success">{JOB_PAGE_COPY.header.ready}</Badge>
          <span className="text-caption text-ash-gray">
            {count} {JOB_PAGE_COPY.header.uniqueTagsSuffix}
          </span>
        </div>
      </div>
      <a href={csvHref} download>
        <Button size="lg">
          <Download />
          {JOB_PAGE_COPY.header.downloadCsv}
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
  detect: JOB_PAGE_COPY.processing.detect,
  finalize: JOB_PAGE_COPY.processing.finalize,
  dedup: JOB_PAGE_COPY.processing.dedup,
  done: JOB_PAGE_COPY.processing.done,
};

function Processing({ job }: { job: Job | null }) {
  const progress = job?.progress ?? 0;
  const phase =
    !job || job.status === "queued"
      ? JOB_PAGE_COPY.processing.queued
      : (job.phase && PHASE_LABEL[job.phase]) ||
        (progress < 0.5
          ? JOB_PAGE_COPY.processing.detect
          : progress < 0.9
            ? JOB_PAGE_COPY.processing.fallbackMid
            : JOB_PAGE_COPY.processing.fallbackFinal);
  return (
    <div className="grid place-items-center py-24">
      <Card feature className="w-full max-w-md p-8 text-center">
        <div className="mx-auto grid size-12 place-items-center rounded-card bg-chartwell-blue/10 text-chartwell-blue">
          <Spinner className="size-6" />
        </div>
        <h1 className="mt-5 font-display text-heading font-medium text-slate-text">
          {JOB_PAGE_COPY.processing.title}
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
          {JOB_PAGE_COPY.processing.hint}
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
          {JOB_PAGE_COPY.errors.genericTitle}
        </h1>
        <p className="mt-2 text-[14px] text-ash-gray">{message}</p>
        <Link to="/">
          <Button className="mt-6">{JOB_PAGE_COPY.errors.backUpload}</Button>
        </Link>
      </Card>
    </div>
  );
}
