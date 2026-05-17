import { useCallback, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { FileVideo, ScanLine, Tag, Table2, UploadCloud, X } from "lucide-react";
import { jobsApi } from "@/api/jobs";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Spinner } from "@/components/ui/spinner";
import { cn, formatBytes } from "@/lib/utils";

const STEPS = [
  {
    icon: ScanLine,
    title: "Детекция",
    body: "Кадры стабилизируются и поворачиваются, на полке находятся ценники.",
  },
  {
    icon: Tag,
    title: "Распознавание",
    body: "Лучший кадр на ценник: цена, штрихкод, QR и поля раскладываются по схеме.",
  },
  {
    icon: Table2,
    title: "Выгрузка",
    body: "Один уникальный ценник — одна строка. 29-колоночный CSV для оценки.",
  },
];

export default function UploadPage() {
  const navigate = useNavigate();
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [submitting, setSubmitting] = useState(false);

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
      const job = await jobsApi.create(file);
      navigate(`/jobs/${job.id}`);
    } catch (err) {
      setSubmitting(false);
      toast.error("Не удалось запустить обработку", {
        description: err instanceof Error ? err.message : "Проверьте, что бэкенд запущен.",
      });
    }
  }, [file, submitting, navigate]);

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
          Загрузите видео проезда робота вдоль полки — система найдёт каждый
          ценник, распознает поля и соберёт выгрузку. Видео, таймкод, кроп и
          разобранные данные — всё на одном экране.
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
              <span className="grid size-12 place-items-center rounded-card bg-chartwell-blue/10 text-chartwell-blue">
                <FileVideo className="size-6" />
              </span>
              <div>
                <p className="text-[15px] font-medium text-slate-text">{file.name}</p>
                <p className="mt-0.5 text-caption text-ash-gray">{formatBytes(file.size)}</p>
              </div>
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  setFile(null);
                  if (inputRef.current) inputRef.current.value = "";
                }}
                className="inline-flex items-center gap-1 text-caption text-ash-gray hover:text-slate-text"
              >
                <X className="size-3.5" /> Выбрать другое видео
              </button>
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

        <div className="flex items-center justify-between gap-4 px-4 py-4">
          <p className="text-caption text-ash-gray">
            Обработка идёт минутами — прогресс будет виден на следующем экране.
          </p>
          <Button size="lg" disabled={!file || submitting} onClick={submit}>
            {submitting ? <Spinner /> : <UploadCloud />}
            {submitting ? "Запуск…" : "Обработать видео"}
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
