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

// ─────────────────────────────────────────────────────────────────────────
// The real pipeline, stage by stage. Faithful to docs/pipeline-reference.md,
// docs/architecture.md, configs/balanced.yaml and catalog-reconciliation.md
// — this page is the jury's guided tour of how one shelf video becomes one
// graded CSV.
// ─────────────────────────────────────────────────────────────────────────
interface Stage {
  icon: LucideIcon;
  title: string;
  tagline: string;
  detail: string;
  techniques: string[];
  knobs?: string[];
  ref?: string;
}

const STAGES: Stage[] = [
  {
    icon: UploadCloud,
    title: "Загрузка видео",
    tagline: "Проезд робота вдоль полки → сервис",
    detail:
      "Видео загружается через SPA в gateway, который кладёт его в общий том и ставит задачу. Обработка идёт минутами, поэтому статус и прогресс опрашиваются отдельным API — пользователь видит живой прогресс-бар, а не зависший экран.",
    techniques: ["Shared-volume handoff", "Job + progress API"],
    ref: "architecture.md §5",
  },
  {
    icon: ScanSearch,
    title: "Детекция ценников",
    tagline: "OpenFoodFacts YOLO11x, дообученный",
    detail:
      "Детектор ценников — зафиксированная база OpenFoodFacts price-tag YOLO11x, которую мы дообучаем (не учим с нуля). Это даёт устойчивые рамки на плотной полке при движущейся камере.",
    techniques: ["YOLO11x · OpenFoodFacts", "Fine-tune (фикс. база)"],
    knobs: ["detector.conf: 0.25", "detector.iou: 0.50"],
    ref: "experiments/finetune_openfoodfacts.yaml",
  },
  {
    icon: Spline,
    title: "Трекинг",
    tagline: "BoT-SORT c компенсацией движения камеры",
    detail:
      "Один физический ценник в кадрах должен стать одним треком. BoT-SORT с camera-motion compensation специально держит ID при едущем роботе; ByteTrack — быстрый профиль. Трекинг по детекциям не требует ручной MOT-разметки.",
    techniques: ["BoT-SORT + CMC", "ByteTrack (fast)"],
    knobs: ["tracker: botsort.yaml", "new_track_thresh: 0.82"],
    ref: "configs/balanced.yaml",
  },
  {
    icon: Aperture,
    title: "Лучший кадр",
    tagline: "Самые резкие кропы на трек",
    detail:
      "Текст читается только на чётких кадрах. На каждый трек выбираются top-k самых резких кропов по метрике Тененграда — дальше распознаём именно их, а не случайный кадр.",
    techniques: ["Tenengrad sharpness", "Top-k crops / track"],
    knobs: ["top_k_crops_per_track: 5", "min_sharpness: 35"],
    ref: "configs/balanced.yaml",
  },
  {
    icon: Crop,
    title: "Ректификация",
    tagline: "Паддинг + CLAHE, цвет сохраняем",
    detail:
      "Кроп берётся с паддингом, CLAHE поднимает локальный контраст по яркости (цвет не трогаем — он несёт тип ценника), вертикальные ценники доворачиваются. Так VLM/OCR видит ровный читаемый ярлык.",
    techniques: ["CLAHE (luminance)", "Padded crop", "Keep colour"],
    knobs: ["rectifier.padding_ratio: 0.10"],
    ref: "configs/balanced.yaml",
  },
  {
    icon: QrCode,
    title: "Считывание кодов",
    tagline: "QR → штрихкод → умный OCR",
    detail:
      "Цепочка декодеров: сначала коды (zxing-cpp + ZBar). Коды на ценниках Ленты — смесь: QR-url, DataMatrix, GS1 DataBar, Code128. QR не короткое замыкание — каждое чтение это голос, который потом сводит агрегатор (политика qr_first_fill_gaps).",
    techniques: ["zxing-cpp", "ZBar", "QR→barcode→OCR"],
    knobs: ["recognition.enable_qr / enable_barcode"],
    ref: "recognition-pipeline.md",
  },
  {
    icon: Languages,
    title: "OCR / VLM",
    tagline: "Qwen3-VL-4B · локально, без облака",
    detail:
      "Структурные поля снимает Qwen3-VL-4B — победитель нашего OCR-бенча, ~8.5 ГБ VRAM, влезает в 4070 Ti. Русский, температура 0. Классический PaddleOCR — лёгкий фолбэк. Всё локально: облачные API на инференсе запрещены правилами хакатона.",
    techniques: ["Qwen3-VL-4B", "Локально · без облака", "PaddleOCR fallback"],
    knobs: ["ocr.backend: qwen3_vl", "vlm_temperature: 0.0"],
    ref: "pipeline-reference.md",
  },
  {
    icon: ListChecks,
    title: "Парсер",
    tagline: "Текст → поля; «нет» ≠ пусто",
    detail:
      "Сырой текст раскладывается в структурированные поля: русские форматы цен, размер скидки (<100₽ → %, ≥100₽ → ₽). Ключевое различие: «нет» (поля нет на ценнике) и пусто (есть, но не распознано) — их путаница стоит баллов.",
    techniques: ["Russian price formats", "«нет» ≠ пусто"],
    ref: "task.md §3.3 / §5.3",
  },
  {
    icon: Boxes,
    title: "Агрегация и дедуп",
    tagline: "Голосование по треку + кросс-трек дедуп",
    detail:
      "По каждому треку поля выбираются голосованием всех наблюдений; затем кросс-трек дедуп по IoU + времени + содержимому/штрихкоду схлопывает повторы. На выходе — один уникальный ценник. Дубликаты дорого штрафуются метрикой.",
    techniques: ["Per-field voting", "Cross-track dedup"],
    knobs: ["dedup_iou_threshold: 0.4", "min_observations: 2"],
    ref: "pipeline-reference.md",
  },
  {
    icon: Library,
    title: "Сверка с каталогом",
    tagline: "OCR против мастер-каталога Ленты (~625k)",
    detail:
      "Распознанные штрихкод и название сверяются с мастер-каталогом Ленты (db_hack.csv, ~625k позиций, локально, без сети). Штрихкод — первичный ключ: нашёлся в каталоге → каноничное чистое имя (заполнить пустое / опц. исправить мис-рид). Пропавший штрихкод восстанавливается по уверенному нечёткому совпадению имени, мис-рид ловится контрольной цифрой GS1. «нет» никогда не перетирается. Покрытие штрихкод→каталог ~97% на разметке.",
    techniques: [
      "Мастер-каталог Ленты",
      "RapidFuzz token_set_ratio",
      "GS1 check digit",
      "Локально · без сети",
    ],
    knobs: ["catalog: real_data/db_hack.csv", "name-policy: fill (safe)"],
    ref: "catalog-reconciliation.md",
  },
  {
    icon: FileCheck2,
    title: "Сабмит",
    tagline: "29-колоночный CSV · один ценник = одна строка",
    detail:
      "Единый контракт всего продукта — 29-колоночный CSV (UTF-8, запятая, точка-десятичная, QUOTE_MINIMAL). Один владелец-производитель, gateway и SPA держат схему в lock-step. Это то, что превращает три сервиса в один продукт.",
    techniques: ["29 колонок", "UTF-8 · QUOTE_MINIMAL"],
    ref: "architecture.md §5.6",
  },
];

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
          Lenta Tech Life 2026 · Пайплайн
        </p>
        <h1 className="mt-4 font-display text-heading-lg font-medium text-slate-text sm:text-display">
          Путь видео до сабмита
        </h1>
        <p className="mx-auto mt-4 max-w-xl text-[16px] leading-[1.6] text-ash-gray">
          Одиннадцать шагов от сырого проезда робота вдоль полки до
          29-колоночного CSV. Нажмите «Прогнать видео» — и проследите за
          кадром на каждом этапе, или кликните любой шаг.
        </p>
        <div className="mt-6 flex flex-wrap items-center justify-center gap-3">
          <Button size="lg" onClick={onPlay} aria-label="Запустить прогон">
            {playing ? <Pause /> : <Play />}
            {playing
              ? "Пауза"
              : active >= last
                ? "Прогнать заново"
                : active <= 0
                  ? "Прогнать видео"
                  : "Продолжить"}
          </Button>
          <Button variant="ghost" size="lg" onClick={reset} aria-label="Сбросить">
            <RotateCcw /> Сброс
          </Button>
        </div>
        <div className="mt-6 flex flex-wrap items-center justify-center gap-2">
          <Badge variant="accent">
            <Sparkles className="size-3" /> Полностью локально · без облачных API
          </Badge>
          <Badge variant="info">GPU · Qwen3-VL-4B</Badge>
          <Badge variant="neutral">29-колоночный контракт</Badge>
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
                          {isActive ? "Свернуть" : "Подробнее"}
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
            Один продукт
          </p>
          <p className="max-w-xl text-[14px] leading-[1.65] text-ash-gray">
            Детектор, трекинг, распознавание и каталог-сверка держатся на
            одном контракте — 29 колонок CSV. Тот же байт-формат у пайплайна,
            gateway и фронтенда: меняешь одного владельца — меняешь всех трёх.
          </p>
        </Card>
      </section>
    </div>
  );
}
