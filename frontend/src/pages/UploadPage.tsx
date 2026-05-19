import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import {
  FileVideo,
  RotateCw,
  ScanLine,
  Tag,
  Table2,
  UploadCloud,
  X,
} from "lucide-react";
import { jobsApi, type Rotation } from "@/api/jobs";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Spinner } from "@/components/ui/spinner";
import { UPLOAD_PAGE_COPY } from "@/content/uploadPageContent";
import { cn, formatBytes } from "@/lib/utils";

const STEPS = [
  {
    icon: ScanLine,
    ...UPLOAD_PAGE_COPY.steps[0],
  },
  {
    icon: Tag,
    ...UPLOAD_PAGE_COPY.steps[1],
  },
  {
    icon: Table2,
    ...UPLOAD_PAGE_COPY.steps[2],
  },
];

const ROTATE_NEXT: Record<Rotation, Rotation> = {
  ccw: "none",
  none: "cw",
  cw: "ccw",
};
// CSS preview = how the detector will see the frames.
const ROTATE_DEG: Record<Rotation, number> = { none: 0, ccw: -90, cw: 90 };
const ROTATE_LABEL: Record<Rotation, string> = UPLOAD_PAGE_COPY.rotateLabels;
const DEFAULT_ROTATION: Rotation = "ccw";

export default function UploadPage() {
  const navigate = useNavigate();
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  // Detector-only pre-rotation. Robot shelf videos are camera-sideways by
  // default, matching balanced.yaml's ccw detector path. This never changes
  // the stored video, review playback or graded CSV coords.
  const [rotation, setRotation] = useState<Rotation>(DEFAULT_ROTATION);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  // Own the object URL for the local preview; revoke when it changes/clears.
  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const pick = useCallback((f: File | undefined | null) => {
    if (!f) return;
    if (!f.type.startsWith("video/")) {
      toast.error(UPLOAD_PAGE_COPY.toasts.invalidFileTitle, {
        description: UPLOAD_PAGE_COPY.toasts.invalidFileDescription,
      });
      return;
    }
    setFile(f);
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      pick(e.dataTransfer.files?.[0]);
    },
    [pick],
  );

  const submit = useCallback(async () => {
    if (!file || submitting) return;
    setSubmitting(true);
    try {
      const job = await jobsApi.create(file, rotation);
      navigate(`/jobs/${job.id}`);
    } catch (err) {
      setSubmitting(false);
      toast.error(UPLOAD_PAGE_COPY.toasts.submitFailedTitle, {
        description:
          err instanceof Error
            ? err.message
            : UPLOAD_PAGE_COPY.toasts.submitFailedFallbackDescription,
      });
    }
  }, [file, submitting, navigate, rotation]);

  return (
    <div className="flex flex-col gap-12">
      {/* Hero */}
      <section className="mx-auto max-w-2xl pt-6 text-center">
        <p className="text-caption font-medium uppercase tracking-[0.14em] text-chartwell-blue">
          {UPLOAD_PAGE_COPY.hero.kicker}
        </p>
        <h1 className="mt-4 font-display text-heading-lg font-medium text-slate-text sm:text-display">
          {UPLOAD_PAGE_COPY.hero.title}
        </h1>
        <p className="mx-auto mt-4 max-w-xl text-[16px] leading-[1.6] text-ash-gray">
          {UPLOAD_PAGE_COPY.hero.lead}
        </p>
      </section>

      {/* Upload */}
      <Card feature className="mx-auto w-full max-w-2xl p-2">
        <div
          role="button"
          tabIndex={0}
          aria-label={UPLOAD_PAGE_COPY.uploadCard.ariaUpload}
          onClick={() => !file && inputRef.current?.click()}
          onKeyDown={(e) => {
            if ((e.key === "Enter" || e.key === " ") && !file) {
              e.preventDefault();
              inputRef.current?.click();
            }
          }}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
          className={cn(
            "rounded-card border border-dashed px-6 py-14 text-center transition-colors",
            file
              ? "border-stone-border bg-canvas-fog/60"
              : "cursor-pointer border-platinum-outline hover:border-chartwell-blue hover:bg-sky-tint/20",
            dragOver && "border-chartwell-blue bg-sky-tint/30",
          )}
        >
          <input
            ref={inputRef}
            type="file"
            accept="video/*"
            className="hidden"
            onChange={(e) => pick(e.target.files?.[0])}
          />

          {file ? (
            <div className="flex flex-col items-center gap-4">
              {/* Local preview — judge orientation before uploading. */}
              <div className="grid h-72 w-full place-items-center overflow-hidden rounded-input bg-ghost-ink">
                {previewUrl ? (
                  <video
                    key={previewUrl}
                    src={previewUrl}
                    muted
                    controls
                    playsInline
                    preload="metadata"
                    className="max-h-full max-w-full object-contain"
                    style={{
                      transform: `rotate(${ROTATE_DEG[rotation]}deg)`,
                      transition: "transform 200ms ease",
                    }}
                  />
                ) : (
                  <FileVideo className="size-8 text-ash-gray" />
                )}
              </div>

              <div className="text-center">
                <p className="text-[15px] font-medium text-slate-text">{file.name}</p>
                <p className="mt-0.5 text-caption text-ash-gray">{formatBytes(file.size)}</p>
              </div>

              <div className="flex flex-wrap items-center justify-center gap-2">
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    setRotation((r) => ROTATE_NEXT[r]);
                  }}
                  className="inline-flex items-center gap-1.5 rounded-pill border border-stone-border bg-cloud-white px-3 py-1.5 text-caption font-medium text-slate-text hover:border-chartwell-blue hover:text-chartwell-blue"
                >
                  <RotateCw className="size-3.5" /> {UPLOAD_PAGE_COPY.uploadCard.rotateButtonPrefix} · {ROTATE_LABEL[rotation]}
                </button>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    setFile(null);
                    setRotation(DEFAULT_ROTATION);
                    if (inputRef.current) inputRef.current.value = "";
                  }}
                  className="inline-flex items-center gap-1 rounded-pill px-3 py-1.5 text-caption text-ash-gray hover:text-slate-text"
                >
                  <X className="size-3.5" /> {UPLOAD_PAGE_COPY.uploadCard.otherVideo}
                </button>
              </div>

              <p className="max-w-md text-center text-caption text-ash-gray">
                {UPLOAD_PAGE_COPY.uploadCard.rotateHint}
              </p>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-4">
              <span className="grid size-12 place-items-center rounded-card bg-cloud-white text-ash-gray shadow-subtle">
                <UploadCloud className="size-6" />
              </span>
              <div>
                <p className="text-[15px] font-medium text-slate-text">
                  {UPLOAD_PAGE_COPY.uploadCard.dropTitle}
                </p>
                <p className="mt-1 text-caption text-ash-gray">
                  {UPLOAD_PAGE_COPY.uploadCard.dropHint}
                </p>
              </div>
            </div>
          )}
        </div>

        <div className="flex items-center justify-between gap-4 px-4 py-4">
          <p className="text-caption text-ash-gray">
            {UPLOAD_PAGE_COPY.uploadCard.processHint}
          </p>
          <Button size="lg" disabled={!file || submitting} onClick={submit}>
            {submitting ? <Spinner /> : <UploadCloud />}
            {submitting ? UPLOAD_PAGE_COPY.uploadCard.submitting : UPLOAD_PAGE_COPY.uploadCard.submit}
          </Button>
        </div>
      </Card>

      {/* How it works */}
      <section className="mx-auto grid w-full max-w-4xl gap-4 sm:grid-cols-3">
        {STEPS.map(({ icon: Icon, title, body }, i) => (
          <Card key={title} className="p-6">
            <div className="flex items-center gap-2">
              <span className="grid size-7 place-items-center rounded-input bg-chartwell-blue/10 text-chartwell-blue">
                <Icon className="size-4" />
              </span>
              <span className="text-caption font-medium text-ash-gray">
                {UPLOAD_PAGE_COPY.stepLabelPrefix} {i + 1}
              </span>
            </div>
            <h3 className="mt-3 font-display text-heading-sm font-medium text-slate-text">
              {title}
            </h3>
            <p className="mt-1.5 text-[13px] leading-[1.6] text-ash-gray">{body}</p>
          </Card>
        ))}
      </section>
    </div>
  );
}
