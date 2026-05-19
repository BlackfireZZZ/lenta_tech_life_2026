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
  Zap,
} from "lucide-react";
import { jobsApi, type ProcessingMode, type Rotation } from "@/api/jobs";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Spinner } from "@/components/ui/spinner";
import { ServerLimitNote } from "@/components/ServerLimitNote";
import { cn, formatBytes } from "@/lib/utils";

const STEPS = [
  {
    icon: ScanLine,
    title: "Находим ценники",
    body: "Система просматривает видео кадр за кадром и обводит рамкой каждый ценник на полке.",
  },
  {
    icon: Tag,
    title: "Читаем, что на них написано",
    body: "Для каждого ценника берётся самый чёткий кадр — с него считываются цена, скидка, штрихкод и QR-код.",
  },
  {
    icon: Table2,
    title: "Собираем таблицу",
    body: "Один и тот же ценник встречался во многих кадрах — мы сводим его в одну строку. На выходе — готовая таблица, которую можно скачать.",
  },
];

const ROTATE_NEXT: Record<Rotation, Rotation> = {
  none: "ccw",
  ccw: "cw",
  cw: "none",
};
// CSS preview = how the detector will see the frames.
const ROTATE_DEG: Record<Rotation, number> = { none: 0, ccw: -90, cw: 90 };
const ROTATE_LABEL: Record<Rotation, string> = {
  none: "как загружено",
  ccw: "против часовой 90°",
  cw: "по часовой 90°",
};

export default function UploadPage() {
  const navigate = useNavigate();
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  // Detector-only pre-rotation. Default "none" — the upload is trusted
  // as-is; this button is the explicit fix for a sideways clip. It NEVER
  // changes the stored video, the review playback or the graded CSV
  // coords — only how the detector model sees frames.
  const [rotation, setRotation] = useState<Rotation>("none");
  // Recognition depth. Default = full (unchecked): the heavy text model
  // reads several shots per tag and votes. Checked = fast: one shot per
  // tag, much quicker. Detection is the same in both — it is already fast.
  const [fast, setFast] = useState(false);
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
      toast.error("Нужен видеофайл", {
        description: "Загрузите запись с полки (.mp4, .mov, .mkv …).",
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
      const mode: ProcessingMode = fast ? "fast" : "full";
      const job = await jobsApi.create(file, rotation, mode);
      navigate(`/jobs/${job.id}`);
    } catch (err) {
      setSubmitting(false);
      toast.error("Не удалось запустить обработку", {
        description: err instanceof Error ? err.message : "Проверьте, что сервер запущен.",
      });
    }
  }, [file, submitting, navigate, rotation, fast]);

  return (
    <div className="flex flex-col gap-12">
      {/* Hero */}
      <section className="mx-auto max-w-2xl pt-6 text-center">
        <p className="text-caption font-medium uppercase tracking-[0.14em] text-chartwell-blue">
          Lenta Tech Life 2026
        </p>
        <h1 className="mt-4 font-display text-heading-lg font-medium text-slate-text sm:text-display">
          Полка под контролем
        </h1>
        <p className="mx-auto mt-4 max-w-xl text-[16px] leading-[1.6] text-ash-gray">
          Загрузите видео, на котором робот едет вдоль полки, — система найдёт
          каждый ценник, прочитает, что на нём написано, и соберёт всё в одну
          таблицу. Видео, момент в кадре, увеличенный ярлык и распознанные
          данные — всё на одном экране.
        </p>
      </section>

      {/* Upload */}
      <Card feature className="mx-auto w-full max-w-2xl p-2">
        <div
          role="button"
          tabIndex={0}
          aria-label="Загрузить видео"
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
                  <RotateCw className="size-3.5" /> Повернуть · {ROTATE_LABEL[rotation]}
                </button>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    setFile(null);
                    setRotation("none");
                    if (inputRef.current) inputRef.current.value = "";
                  }}
                  className="inline-flex items-center gap-1 rounded-pill px-3 py-1.5 text-caption text-ash-gray hover:text-slate-text"
                >
                  <X className="size-3.5" /> Другое видео
                </button>
              </div>

              <p className="max-w-md text-center text-caption text-ash-gray">
                Поворот нужен, только если ценники лежат на боку, — он влияет
                лишь на то, как кадры «видит» распознавание. Сам файл и видео
                в результатах останутся как есть.
              </p>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-4">
              <span className="grid size-12 place-items-center rounded-card bg-cloud-white text-ash-gray shadow-subtle">
                <UploadCloud className="size-6" />
              </span>
              <div>
                <p className="text-[15px] font-medium text-slate-text">
                  Перетащите видео сюда или нажмите, чтобы выбрать
                </p>
                <p className="mt-1 text-caption text-ash-gray">
                  MP4 / MOV / MKV · запись проезда робота вдоль полки
                </p>
              </div>
            </div>
          )}
        </div>

        <div className="flex flex-col gap-3 px-4 py-4">
          {/* Fast vs full recognition. Detection is NOT cut here (it is
              already fast — the user asked to keep it); only the slow
              per-tag text recognition runs on fewer shots when fast is on. */}
          <label
            className={cn(
              "flex cursor-pointer items-start gap-3 rounded-card border px-4 py-3 transition-colors",
              fast
                ? "border-chartwell-blue bg-sky-tint/30"
                : "border-stone-border bg-canvas-fog hover:border-chartwell-blue/60",
            )}
          >
            <input
              type="checkbox"
              checked={fast}
              onChange={(e) => setFast(e.target.checked)}
              className="mt-0.5 size-4 shrink-0 accent-chartwell-blue"
            />
            <span className="min-w-0">
              <span className="flex items-center gap-1.5 text-[14px] font-medium text-slate-text">
                <Zap className="size-4 text-chartwell-blue" /> Быстрый прогон
              </span>
              <span className="mt-0.5 block text-caption leading-[1.55] text-ash-gray">
                Чтение текста с ценников — самая долгая часть. В быстром
                режиме нейросеть распознаёт каждый ценник по одному лучшему
                кадру, а не по нескольким: заметно быстрее, точность чуть
                ниже. Поиск ценников в кадре не меняется — он и так быстрый.
              </span>
            </span>
          </label>

          <div className="flex items-center justify-between gap-4">
            <p className="text-caption text-ash-gray">
              {fast
                ? "Быстрый режим: одна попытка распознавания на ценник."
                : "Полный режим: несколько кадров на ценник и голосование за точность."}
            </p>
            <Button size="lg" disabled={!file || submitting} onClick={submit}>
              {submitting ? <Spinner /> : <UploadCloud />}
              {submitting ? "Запуск…" : "Обработать видео"}
            </Button>
          </div>
        </div>
      </Card>

      {/* Why a video may wait — a real limit of the cheap rented server,
          stated up front so the queue screen is no surprise. */}
      <ServerLimitNote className="mx-auto w-full max-w-2xl" />

      {/* How it works */}
      <section className="mx-auto grid w-full max-w-4xl gap-4 sm:grid-cols-3">
        {STEPS.map(({ icon: Icon, title, body }, i) => (
          <Card key={title} className="p-6">
            <div className="flex items-center gap-2">
              <span className="grid size-7 place-items-center rounded-input bg-chartwell-blue/10 text-chartwell-blue">
                <Icon className="size-4" />
              </span>
              <span className="text-caption font-medium text-ash-gray">Шаг {i + 1}</span>
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
