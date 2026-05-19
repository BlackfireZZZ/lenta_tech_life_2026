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
    tagline: "Вы загружаете видео — оно встаёт в очередь",
    detail:
      "Вы загружаете видео на сайт. Файл уходит на сервер и встаёт в очередь на обработку. Распознавание длится несколько минут, поэтому вместо «зависшего» экрана вы всё время видите живой прогресс-бар — он показывает, какой шаг идёт прямо сейчас.",
    techniques: ["Очередь задач", "Живой прогресс"],
    ref: "architecture.md §5",
  },
  {
    icon: ScanSearch,
    title: "Поиск ценников в кадре",
    tagline: "Нейросеть обводит каждый ценник на полке",
    detail:
      "Система просматривает каждый кадр и обводит рамкой каждый ценник на полке. В основе — готовая нейросеть, которую обучили узнавать ценники в открытом проекте OpenFoodFacts; мы дообучили её на полках Ленты, а не учили с нуля. Поэтому рамки получаются устойчивыми даже на плотно заставленной полке и при движущейся камере.",
    techniques: ["Нейросеть YOLO11x", "Дообучена на полках Ленты"],
    knobs: ["detector.conf: 0.25", "detector.iou: 0.50"],
    ref: "experiments/finetune_openfoodfacts.yaml",
  },
  {
    icon: Spline,
    title: "Ведение ценника по кадрам",
    tagline: "Узнаём один и тот же ценник в разных кадрах",
    detail:
      "Робот едет — и один и тот же ценник попадает в десятки кадров. Чтобы потом не посчитать его много раз, система «ведёт» каждый ценник от кадра к кадру и понимает, что это всё тот же ярлык (это называют трекингом). Алгоритм специально учитывает, что камера движется, — иначе на ходу ценники легко перепутать.",
    techniques: ["Трекинг BoT-SORT", "Учёт движения камеры"],
    knobs: ["tracker: botsort.yaml", "new_track_thresh: 0.82"],
    ref: "configs/balanced.yaml",
  },
  {
    icon: Aperture,
    title: "Выбор лучшего кадра",
    tagline: "Берём самые чёткие кадры каждого ценника",
    detail:
      "Пока ценник проезжает мимо камеры, часть кадров получается смазанной. Из всех кадров с одним ценником система отбирает несколько самых резких и вырезает с них только сам ярлык (такой вырезанный кусочек изображения мы называем кроп). Дальше читаем именно эти чёткие кропы, а не случайный смазанный кадр.",
    techniques: ["Оценка резкости", "Несколько лучших кадров на ценник"],
    knobs: ["top_k_crops_per_track: 5", "min_sharpness: 35"],
    ref: "configs/balanced.yaml",
  },
  {
    icon: Crop,
    title: "Подготовка ярлыка",
    tagline: "Выравниваем и делаем ярлык читаемым",
    detail:
      "Вырезанный ярлык немного «причёсываем»: добавляем поля по краям, чтобы ничего не обрезалось, аккуратно поднимаем контраст (цвет при этом не трогаем — по цвету видно тип ценника), наклонённые ярлыки доворачиваем. Так распознавание получает ровную и чёткую картинку.",
    techniques: ["Поднятие контраста", "Поля по краям", "Цвет сохраняем"],
    knobs: ["rectifier.padding_ratio: 0.10"],
    ref: "configs/balanced.yaml",
  },
  {
    icon: QrCode,
    title: "Считывание кода",
    tagline: "Сначала пробуем прочитать код на ценнике",
    detail:
      "Сначала пробуем считать сам код на ценнике. На ценниках Ленты они разные: где-то QR-ссылка, где-то квадратный код DataMatrix, где-то обычный полосатый штрихкод. Каждое успешное считывание — это «голос»: финальное значение система выбирает потом из всех голосов, а не по первому попавшемуся.",
    techniques: ["Несколько сканеров кодов", "QR → штрихкод → текст"],
    knobs: ["recognition.enable_qr / enable_barcode"],
    ref: "recognition-pipeline.md",
  },
  {
    icon: Languages,
    title: "Распознавание текста (OCR)",
    tagline: "Нейросеть читает ярлык — локально, без облака",
    detail:
      "OCR — это распознавание текста по картинке. То, что не зашито в код (название, цену, скидку), «считывает» с ярлыка нейросеть, которая умеет одновременно и видеть изображение, и отвечать текстом. Мы сравнили несколько таких моделей и взяли лучшую. Она работает прямо на нашем компьютере — без интернета и облака, этого требуют правила конкурса.",
    techniques: ["Нейросеть Qwen3-VL-4B", "Только локально, без облака", "Запасной PaddleOCR"],
    knobs: ["ocr.backend: qwen3_vl", "vlm_temperature: 0.0"],
    ref: "pipeline-reference.md",
  },
  {
    icon: ListChecks,
    title: "Раскладка по полям",
    tagline: "Превращаем распознанный текст в аккуратные поля",
    detail:
      "Распознанный текст раскладывается по полям: цена в рублях, размер скидки и так далее. Важная тонкость: «нет» (такого поля на ценнике вообще не было) и «пусто» (поле было, но мы его не разобрали) — это разные вещи. Если их перепутать, оценка снижается, поэтому мы их строго различаем.",
    techniques: ["Российские форматы цен", "«нет» ≠ «пусто»"],
    ref: "task.md §3.3 / §5.3",
  },
  {
    icon: Boxes,
    title: "Сводим повторы в один ценник",
    tagline: "Один реальный ценник — ровно одна строка",
    detail:
      "Один ценник мы видели в десятках кадров и каждый раз что-то распознавали. Здесь все наблюдения по нему сводятся вместе голосованием — для каждого поля берётся самый частый ответ. Потом идёт дедупликация (от слова «дубликат» — удаление повторов): если два разных «следа» на самом деле один и тот же ценник, они склеиваются в один. На выходе каждый реальный ценник встречается ровно один раз — за дубликаты в таблице оценка заметно снижается.",
    techniques: ["Голосование по полям", "Склейка дубликатов"],
    knobs: ["dedup_iou_threshold: 0.4", "min_observations: 2"],
    ref: "pipeline-reference.md",
  },
  {
    icon: Library,
    title: "Сверка с каталогом",
    tagline: "Сверяем с каталогом товаров Ленты",
    detail:
      "Распознанные штрихкод и название сверяются с официальным каталогом товаров Ленты (около 625 000 позиций, хранится локально, без интернета). Штрихкод — главный ключ: нашли его в каталоге — подставляем точное, «чистое» название. Если штрихкод не прочитался, его можно восстановить по уверенному совпадению названия, а ошибочно считанный код отсекаем по контрольной цифре. То, чего на ценнике нет, мы при этом никогда не дописываем. В итоге в каталоге находится около 97% ценников с прочитанным штрихкодом.",
    techniques: [
      "Каталог Ленты (~625 000)",
      "Нечёткое сравнение названий",
      "Проверка контрольной цифры",
      "Локально, без сети",
    ],
    knobs: ["catalog: real_data/db_hack.csv", "name-policy: fill (safe)"],
    ref: "catalog-reconciliation.md",
  },
  {
    icon: FileCheck2,
    title: "Итоговая таблица",
    tagline: "Таблица из 29 столбцов: один ценник — одна строка",
    detail:
      "На выходе — таблица из 29 столбцов: одна строка на один уникальный ценник (формат CSV, он открывается в Excel или Google Таблицах). Это и есть итоговый результат, который можно скачать и проверить. Один и тот же формат используют все части системы, поэтому данные нигде не «разъезжаются».",
    techniques: ["29 столбцов", "Формат CSV (UTF-8)"],
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
          Lenta Tech Life 2026 · Как это работает
        </p>
        <h1 className="mt-4 font-display text-heading-lg font-medium text-slate-text sm:text-display">
          Путь видео — от записи до таблицы
        </h1>
        <p className="mx-auto mt-4 max-w-xl text-[16px] leading-[1.6] text-ash-gray">
          Одиннадцать шагов: от видео, где робот едет вдоль полки, до готовой
          таблицы с ценниками. Нажмите «Показать по шагам» — и пройдите весь
          путь по очереди, или откройте любой шаг сами.
        </p>
        <div className="mt-6 flex flex-wrap items-center justify-center gap-3">
          <Button size="lg" onClick={onPlay} aria-label="Запустить прогон">
            {playing ? <Pause /> : <Play />}
            {playing
              ? "Пауза"
              : active >= last
                ? "Показать заново"
                : active <= 0
                  ? "Показать по шагам"
                  : "Продолжить"}
          </Button>
          <Button variant="ghost" size="lg" onClick={reset} aria-label="Сбросить">
            <RotateCcw /> Сброс
          </Button>
        </div>
        <div className="mt-6 flex flex-wrap items-center justify-center gap-2">
          <Badge variant="accent">
            <Sparkles className="size-3" /> Работает локально — без интернета и облака
          </Badge>
          <Badge variant="info">Нейросеть на видеокарте</Badge>
          <Badge variant="neutral">Итог — таблица из 29 столбцов</Badge>
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
            Одно целое
          </p>
          <p className="max-w-xl text-[14px] leading-[1.65] text-ash-gray">
            Поиск ценников, их ведение по кадрам, распознавание и сверка с
            каталогом — это разные части, но все они говорят на одном «языке»:
            таблице из 29 столбцов. Один формат на всю систему — поэтому данные
            нигде не теряются и не путаются.
          </p>
        </Card>
      </section>
    </div>
  );
}
