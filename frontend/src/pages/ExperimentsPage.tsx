import { useState } from "react";
import type { LucideIcon } from "lucide-react";
import {
  Check,
  ExternalLink,
  FlaskConical,
  Languages,
  QrCode,
  ScanSearch,
  SlidersHorizontal,
  Sparkles,
  X,
} from "lucide-react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

// ─────────────────────────────────────────────────────────────────────────
// What we actually tried. Distilled from the detector / OCR / QR campaigns,
// the friend's zero-shot baseline, the tracker/level-2 ablations and the
// parameter sweeps across the worktrees. Jury-facing: breadth of the search
// + the measured conclusions (incl. the rigorous negative results).
// ─────────────────────────────────────────────────────────────────────────
type Verdict = "prod" | "explored" | "rejected";

interface Attempt {
  label: string;
  note?: string;
  verdict: Verdict;
}

interface DatasetRef {
  name: string;
  url: string;
}

interface ExpSection {
  id: string;
  icon: LucideIcon;
  title: string;
  blurb: string;
  stat: string;
  datasets?: DatasetRef[];
  tried: Attempt[];
  finding: string;
  shipped: string;
}

const SECTIONS: ExpSection[] = [
  {
    id: "detector",
    icon: ScanSearch,
    title: "Чем находить ценники",
    blurb:
      "От попытки обучить свою модель с нуля до дообучения уже готовой нейросети на наборах открытых данных.",
    stat: "4 набора данных · 6+ моделей",
    datasets: [
      {
        name: "OpenFoodFacts price-tag-detection",
        url: "https://huggingface.co/datasets/openfoodfacts/price-tag-detection",
      },
      {
        name: "SOVAR Roboflow price-tag v2",
        url: "https://universe.roboflow.com/sovar/price-tag-detection-r5jlv/dataset/2",
      },
      {
        name: "Kaggle Supermarket Shelves",
        url: "https://www.kaggle.com/datasets/humansintheloop/supermarket-shelves-dataset",
      },
      {
        name: "Ultralytics SKU-110K",
        url: "https://docs.ultralytics.com/datasets/detect/sku-110k",
      },
    ],
    tried: [
      { label: "Обучить свою модель с нуля", verdict: "rejected", note: "своих данных слишком мало — не взлетело" },
      {
        label: "Готовая нейросеть OpenFoodFacts (дообучаем, не учим с нуля)",
        verdict: "prod",
        note: "наша основа",
      },
      {
        label: "YOLO-World без обучения — просто по слову «ценник»",
        verdict: "explored",
        note: "быстрая точка отсчёта",
      },
      {
        label: "Перебор разных моделей (RF-DETR, YOLO 26 / 12 / 11 разных размеров)",
        verdict: "explored",
        note: "искали, что точнее",
      },
      {
        label:
          "Разные миксы открытых наборов данных + предобучение на крупном наборе полок",
        verdict: "explored",
        note: "больше внешних примеров",
      },
      {
        label:
          "Искусственно «портим» обучающие кадры под именно эту камеру: смаз, шум, блики, тени, искажение объектива",
        verdict: "prod",
      },
      {
        label: "Учим модель на её же ошибках — собираем то, что она ложно приняла за ценник",
        verdict: "prod",
        note: "меньше ложных срабатываний без потери находок",
      },
      {
        label: "Переворот по вертикали, сильное искажение цвета, перенастройка на крошечном объёме",
        verdict: "rejected",
        note: "ломают цену или ничего не дают",
      },
    ],
    finding:
      "Сильнее всего находимость ценников поднимает разрешение кадра: переход с 1280 до 1536 точек дал примерно +70%. Данные важнее самой модели. Стандартные метрики на таком маленьком наборе обманывают, поэтому мы сверялись по отложенным видео и глазами, а не по цифрам обучения. Параметры камеры мы вычислили сами: маленькая матрица, широкий объектив, сильное «бочкообразное» искажение по краям.",
    shipped:
      "Дообученная нейросеть OpenFoodFacts (только класс «ценник»), кадр от 1536 точек, обучение с искажениями под нашу камеру и дообучение на собственных ошибках.",
  },
  {
    id: "ocr",
    icon: Languages,
    title: "Чем читать текст",
    blurb:
      "Сравнение локальных нейросетей, которые «читают» картинку, и пошаговая доводка запроса к модели и разбора её ответа — всё без облака (правило конкурса).",
    stat: "8+ моделей · ~10 замеренных шагов",
    tried: [
      {
        label: "Нейросеть Qwen3-VL-4B",
        verdict: "prod",
        note: "локально, помещается в обычную игровую видеокарту",
      },
      { label: "GLM-OCR 0.9B", verdict: "rejected", note: "слабая, придумывает текст, путает «нет» и пусто" },
      {
        label: "Целый ряд других моделей (HunyuanOCR, PaddleOCR-VL, dots.ocr, MonkeyOCR, RolmOCR, InternVL3)",
        verdict: "explored",
        note: "сравнили на одном тесте",
      },
      { label: "Классические PaddleOCR и Tesseract", verdict: "prod", note: "лёгкие запасные варианты" },
      {
        label: "Аккуратный разбор ответа: чиним формат, в штрихкоде оставляем только цифры, разводим типы цен, чистим примечания",
        verdict: "prod",
      },
      { label: "Очень подробный запрос к модели (вся схема из 28 полей)", verdict: "rejected", note: "модель просто повторяла схему — стало хуже, откатили" },
      { label: "Растягивание вырезанного ярлыка с 1024 до 1536", verdict: "rejected", note: "в сумме хуже: медленнее, штрихкод читается хуже" },
    ],
    finding:
      "Путь был от «не работает совсем» до почти идеального результата на нашем тесте — примерно за 10 аккуратных шагов. Сработала пара вещей: короткий чёткий запрос к модели плюс предсказуемая чистка её ответа кодом. Длинный подробный запрос только всё ухудшил — его откатили. Мелкие копейки в цене «без карты» — это предел качества исходного видео, а не нашего метода.",
    shipped:
      "Модель Qwen3-VL-4B локально, зафиксированные запрос и разбор ответа, PaddleOCR и Tesseract как запасные варианты.",
  },
  {
    id: "qr",
    icon: QrCode,
    title: "Как считывать коды",
    blurb:
      "Коды на ценниках Ленты — это не один аккуратный «QR с ценами», а смесь разных типов. Мы пробовали считывать их по-разному и честно дошли до предела, который задаёт само качество съёмки.",
    stat: "4 сканера · поиск кода по зонам ценника",
    tried: [
      {
        label: "Сразу несколько движков распознавания кодов в связке",
        verdict: "prod",
      },
      {
        label: "Ищем код там, где он по макету ценника и должен быть (2D-код справа сверху, штрихкод — полосой снизу)",
        verdict: "prod",
        note: "именно это и дало результат",
      },
      {
        label: "Разрешаем только реальные для Ленты типы кодов (QR, DataMatrix, EAN/UPC, Code128, DataBar)",
        verdict: "prod",
        note: "чтобы не ловить чужие форматы по ошибке",
      },
      { label: "Проверка контрольной цифры — отбраковываем неверно считанные коды", verdict: "prod", note: "ошибочный код хуже пустого" },
      { label: "Поиск кода «вслепую» по всему ярлыку с поворотами", verdict: "rejected", note: "ноль попаданий" },
      {
        label: "Собирать код из многих кадров сразу",
        verdict: "rejected",
        note: "0 прироста — дело в смазе на видео, а не в сканере",
      },
    ],
    finding:
      "На деле коды совершенно разные: где-то QR-ссылка, где-то непрозрачный квадратный код, где-то многоэтажный или линейный штрихкод. Единственное, что реально помогло их вскрыть, — искать код там, где он расположен по макету ценника. Сравнение «как было / как стало» показало честный потолок около 9%: дальше всё упирается в качество съёмки (смаз стирает мелкие элементы кода), и лечится это условиями записи и сбором из многих кадров, а не донастройкой сканеров.",
    shipped:
      "Связка сканеров + поиск кода по зонам ценника, проверка контрольной цифры; склейка дубликатов сводит редкие прочитанные коды ровно в одну строку по штрихкоду.",
  },
  {
    id: "tuning",
    icon: SlidersHorizontal,
    title: "Тонкая настройка",
    blurb:
      "Десятки мелких настроек — все проверены замером, а не на глаз. Меняем по одной за раз и оставляем только то, что реально дало плюс.",
    stat: "журнал каждого прогона · по одной настройке",
    tried: [
      { label: "Поиск ценников: разрешение кадра, порог уверенности, способ обучения", verdict: "prod" },
      { label: "Ведение по кадрам: выбор алгоритма, учёт движения камеры, пороги", verdict: "prod" },
      { label: "Склейка дубликатов: одинаковый штрихкод → объединяем; учитываем время и пересечение рамок", verdict: "prod" },
      { label: "Выбор лучшего кадра: резкость, размер ярлыка, уверенность и штраф за край кадра", verdict: "prod" },
      { label: "Подготовка ярлыка: поля по краям, контраст; пороги для итогового ответа", verdict: "prod" },
      { label: "Метод: каждый прогон — запись в журнал, проверка на отложенном видео и глазами", verdict: "prod" },
      { label: "Тонкая перенастройка на маленьком объёме данных", verdict: "rejected", note: "почти ничего не меняет" },
    ],
    finding:
      "Правило простое: одна настройка за раз, оставляем только при измеримом улучшении на отложенных данных. Многие «очевидные» крутилки на деле ничего не дали. Решают разрешение, объём данных и качество съёмки — а не тонкая настройка.",
    shipped:
      "Зафиксированные режимы «быстро / сбалансированно / качество», задокументированные настройки и журнал каждого прогона.",
  },
];

const VERDICT: Record<
  Verdict,
  { label: string; hint: string; icon: LucideIcon; cls: string }
> = {
  prod: {
    label: "в работе",
    hint: "вошло в финальную версию",
    icon: Check,
    cls: "border-chartwell-blue/40 bg-chartwell-blue/10 text-slate-text",
  },
  explored: {
    label: "проверили",
    hint: "замерили, но не пригодилось",
    icon: FlaskConical,
    cls: "border-stone-border bg-canvas-fog text-ash-gray",
  },
  rejected: {
    label: "стало хуже",
    hint: "замер показал — не помогает",
    icon: X,
    cls: "border-stone-border bg-transparent text-steel-gray",
  },
};

const VERDICT_ORDER: Verdict[] = ["prod", "explored", "rejected"];

export default function ExperimentsPage() {
  const [activeId, setActiveId] = useState(SECTIONS[0].id);
  const section = SECTIONS.find((s) => s.id === activeId) ?? SECTIONS[0];
  const ActiveIcon = section.icon;

  return (
    <div className="flex flex-col gap-10">
      {/* Hero */}
      <section className="mx-auto max-w-2xl pt-6 text-center">
        <p className="text-caption font-medium uppercase tracking-[0.14em] text-chartwell-blue">
          Lenta Tech Life 2026 · Что мы пробовали
        </p>
        <h1 className="mt-4 font-display text-heading-lg font-medium text-slate-text sm:text-display">
          Что мы перепробовали
        </h1>
        <p className="mx-auto mt-4 max-w-xl text-[16px] leading-[1.6] text-ash-gray">
          Это не «взяли одну готовую модель и сдали». Мы перебрали много
          вариантов: чем находить ценники, чем читать текст, как считывать коды
          и десятки мелких настроек. Здесь и то, что сработало, и то, что мы
          честно проверили и отбросили, — потому что замер показал, что копать
          дальше смысла нет.
        </p>
        <div className="mt-6 flex flex-wrap items-center justify-center gap-2">
          <Badge variant="accent">
            <Sparkles className="size-3" /> Каждое решение — по замеру, а не на глаз
          </Badge>
          <Badge variant="neutral">Всё работает локально</Badge>
        </div>
      </section>

      {/* Category switcher */}
      <section className="mx-auto w-full max-w-4xl">
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          {SECTIONS.map((s) => {
            const Icon = s.icon;
            const on = s.id === activeId;
            return (
              <button
                key={s.id}
                type="button"
                onClick={() => setActiveId(s.id)}
                aria-pressed={on}
                className={cn(
                  "flex cursor-pointer items-center gap-2 rounded-card border px-4 py-3 text-left transition-all duration-200",
                  on
                    ? "border-chartwell-blue/40 bg-sky-tint/15 shadow-card"
                    : "border-stone-border bg-card hover:-translate-y-0.5 hover:border-chartwell-blue/40 hover:shadow-pop",
                )}
              >
                <span
                  className={cn(
                    "grid size-8 shrink-0 place-items-center rounded-input transition-colors",
                    on
                      ? "bg-chartwell-blue text-cloud-white"
                      : "bg-chartwell-blue/10 text-chartwell-blue",
                  )}
                >
                  <Icon className="size-4" />
                </span>
                <span className="min-w-0">
                  <span className="block truncate text-[14px] font-medium text-slate-text">
                    {s.title}
                  </span>
                  <span className="block truncate text-[11px] text-ash-gray">
                    {s.stat}
                  </span>
                </span>
              </button>
            );
          })}
        </div>

        {/* Legend — what the status pills mean */}
        <div className="mt-3 flex flex-wrap items-center justify-center gap-x-5 gap-y-1.5">
          {VERDICT_ORDER.map((v) => {
            const meta = VERDICT[v];
            const MIcon = meta.icon;
            return (
              <span key={v} className="inline-flex items-center gap-1.5 text-[12px] text-ash-gray">
                <span
                  className={cn(
                    "inline-flex items-center gap-1 rounded-pill border px-2 py-0.5 text-[11px] font-medium",
                    meta.cls,
                  )}
                >
                  <MIcon className="size-3" />
                  {meta.label}
                </span>
                <span className="text-steel-gray">— {meta.hint}</span>
              </span>
            );
          })}
        </div>
      </section>

      {/* Active section */}
      <section className="mx-auto w-full max-w-4xl">
        <Card feature className="p-6 sm:p-8">
          <div className="flex items-start gap-3">
            <span className="grid size-11 shrink-0 place-items-center rounded-card bg-chartwell-blue/10 text-chartwell-blue">
              <ActiveIcon className="size-5" />
            </span>
            <div className="min-w-0">
              <h2 className="font-display text-heading font-medium text-slate-text">
                {section.title}
              </h2>
              <p className="mt-1 text-[14px] leading-[1.6] text-ash-gray">
                {section.blurb}
              </p>
            </div>
          </div>

          {section.datasets && (
            <div className="mt-6">
              <p className="text-caption font-medium uppercase tracking-[0.12em] text-steel-gray">
                Открытые наборы данных, которые мы использовали
              </p>
              <div className="mt-2 flex flex-wrap gap-2">
                {section.datasets.map((d) => (
                  <a
                    key={d.url}
                    href={d.url}
                    target="_blank"
                    rel="noreferrer noopener"
                    className="inline-flex items-center gap-1.5 rounded-pill border border-stone-border bg-canvas-fog px-3 py-1 text-[12px] text-slate-text transition-colors hover:border-chartwell-blue/40 hover:text-chartwell-blue"
                  >
                    {d.name}
                    <ExternalLink className="size-3" />
                  </a>
                ))}
              </div>
            </div>
          )}

          <div className="mt-6">
            <p className="text-caption font-medium uppercase tracking-[0.12em] text-steel-gray">
              Что пробовали и с каким исходом
            </p>
            <ul className="mt-3 flex flex-col gap-2">
              {section.tried.map((a) => {
                const v = VERDICT[a.verdict];
                const VIcon = v.icon;
                return (
                  <li
                    key={a.label}
                    className="flex items-start justify-between gap-3 rounded-input border border-stone-border bg-cloud-white px-3 py-2.5"
                  >
                    <div className="min-w-0">
                      <span
                        className={cn(
                          "text-[13px] text-slate-text",
                          a.verdict === "rejected" && "text-steel-gray",
                        )}
                      >
                        {a.label}
                      </span>
                      {a.note && (
                        <span className="ml-2 text-[12px] text-ash-gray">
                          — {a.note}
                        </span>
                      )}
                    </div>
                    <span
                      className={cn(
                        "inline-flex shrink-0 items-center gap-1 rounded-pill border px-2 py-0.5 text-[11px] font-medium",
                        v.cls,
                      )}
                    >
                      <VIcon className="size-3" />
                      {v.label}
                    </span>
                  </li>
                );
              })}
            </ul>
          </div>

          <div className="mt-6 grid gap-3 sm:grid-cols-2">
            <div className="rounded-card border border-chartwell-blue/30 bg-chartwell-blue/5 p-4">
              <p className="text-caption font-medium uppercase tracking-[0.12em] text-chartwell-blue">
                Ключевой вывод
              </p>
              <p className="mt-1.5 text-[13px] leading-[1.6] text-slate-text">
                {section.finding}
              </p>
            </div>
            <div className="rounded-card border border-stone-border bg-canvas-fog p-4">
              <p className="text-caption font-medium uppercase tracking-[0.12em] text-steel-gray">
                Что в итоговой версии
              </p>
              <p className="mt-1.5 text-[13px] leading-[1.6] text-slate-text">
                {section.shipped}
              </p>
            </div>
          </div>
        </Card>
      </section>

      {/* Closing */}
      <section className="mx-auto w-full max-w-4xl">
        <Card className="flex flex-col items-center gap-2 p-6 text-center">
          <p className="text-caption font-medium uppercase tracking-[0.12em] text-chartwell-blue">
            Метод
          </p>
          <p className="max-w-2xl text-[14px] leading-[1.65] text-ash-gray">
            Каждое направление — это отдельный замеренный эксперимент: меняем
            одно, проверяем на отложенных видео и глазами, оставляем только при
            подтверждённом плюсе. Неудачные попытки мы тоже показываем — именно
            в этом строгость подхода, а не угадайка.
          </p>
        </Card>
      </section>
    </div>
  );
}
