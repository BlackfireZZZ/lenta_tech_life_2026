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
    title: "Детекторы",
    blurb:
      "От обучения своей модели с нуля до дообучения найденной pretrained-сети на сборных внешних датасетах.",
    stat: "4 датасета · 6+ архитектур",
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
      { label: "Своя модель с нуля", verdict: "rejected", note: "данных мало — мимо" },
      {
        label: "OpenFoodFacts price-tag YOLO11x (найденная pretrained-база)",
        verdict: "prod",
        note: "дообучаем, не учим с нуля",
      },
      {
        label: "Zero-shot YOLO-World (open-vocab, без обучения)",
        verdict: "explored",
        note: "быстрый бейзлайн",
      },
      {
        label: "RF-DETR (DINOv2, ICLR’26) · YOLO26 · YOLO12 · YOLO11 n·s·m",
        verdict: "explored",
        note: "свип архитектур",
      },
      {
        label:
          "Композитные внешние миксы (OFF strict/broad · SOVAR clean · HITL Price) + SKU-110K object-pretrain",
        verdict: "explored",
        note: "источник внешнего сигнала",
      },
      {
        label:
          "Камеро-матчевые аугментации под реверс-инженеренный barrel k1≈−0.276 (MotionBlur·Defocus·ISONoise·CLAHE·Shadow/Flare·Perspective·CoarseDropout·distortion-jitter·mosaic-свип)",
        verdict: "prod",
      },
      {
        label: "Hard-negative mining из неразмеченных видео + именованная FP-таксономия",
        verdict: "prod",
        note: "точность без потери recall",
      },
      {
        label: "VerticalFlip · агрессивный HSV · перетюн на малом объёме",
        verdict: "rejected",
        note: "ломают сигнал цены / инертно",
      },
    ],
    finding:
      "Разрешение — главный рычаг recall (1280→1536 ≈ +70% относительно). Данные важнее архитектуры; mAP врёт на таком объёме — отбор по held-out видео, визуальному QA и логу каждого рана, не по loss/mAP. Камера реверс-инженерена: 1/2.8″, 2.8 мм, сильный barrel k1≈−0.276.",
    shipped:
      "Дообученный OpenFoodFacts YOLO11x, один класс price_tag, imgsz ≥ 1536, камеро-сопоставленные аугментации, hard-negative loop.",
  },
  {
    id: "ocr",
    icon: Languages,
    title: "OCR / VLM",
    blurb:
      "Бейк-офф локальных VLM и измеренная итерация промпта и парсера — без облака (правило хакатона).",
    stat: "8+ движков · ~10 измеренных итераций",
    tried: [
      {
        label: "Qwen3-VL-4B",
        verdict: "prod",
        note: "локально, ~8.5 ГБ VRAM, влезает в 4070 Ti",
      },
      { label: "GLM-OCR 0.9B", verdict: "rejected", note: "слабая, галлюцинации, игнорит «нет»" },
      {
        label: "HunyuanOCR 1B · PaddleOCR-VL 1.5 · dots.ocr · MonkeyOCR · RolmOCR · InternVL3",
        verdict: "explored",
        note: "vLLM-матрица бейк-офф",
      },
      { label: "Классический PaddleOCR + Tesseract", verdict: "prod", note: "лёгкие фолбэки" },
      {
        label: "Надёжный JSON-repair + digits-only barcode + threshold-цены + sanitizer additional_info",
        verdict: "prod",
      },
      { label: "Verbose-промпт (28-key schema)", verdict: "rejected", note: "schema-echo, регресс — откатили" },
      { label: "Апскейл кропа 1024→1536", verdict: "rejected", note: "net-negative: медленнее, barcode хуже" },
    ],
    finding:
      "Путь: сломано (0.0) → near-perfect на измеряемом бенче за ~10 контролируемых итераций. Ключ — минимальный промпт + детерминированные фиксы парсера; многословный промпт регрессировал и откачен. Мелкие копейки «без карты» — потолок разрешения источника, не метода.",
    shipped:
      "Qwen3-VL-4B локально, залоченные промпт+парсер, PaddleOCR/Tesseract как фолбэки, vLLM-путь для любой модели.",
  },
  {
    id: "qr",
    icon: QrCode,
    title: "QR / штрихкоды",
    blurb:
      "Коды Ленты — не «QR с ценами», а смесь символик. Пробовали сканить разными путями и честно дошли до предела входа.",
    stat: "4 декодера · layout-ROI каскад",
    tried: [
      {
        label: "Полный стек декодеров: zxing-cpp + ZBar + OpenCV WeChat + pylibdmtx",
        verdict: "prod",
      },
      {
        label: "Doc-informed layout-ROI каскад (2D сверху-справа, ШК нижней полосой)",
        verdict: "prod",
        note: "именно это вскрыло коды",
      },
      {
        label: "Ограничение символик: QR/DataMatrix/EAN/UPC/Code128/DataBar",
        verdict: "prod",
        note: "precision-guard от ITF/Codabar",
      },
      { label: "GTIN-checksum отбраковка мис-ридов", verdict: "prod", note: "неверный код хуже пустого — P0-ключ" },
      { label: "Слепая градиентная локализация + повороты", verdict: "rejected", note: "ноль попаданий" },
      {
        label: "Level-2: широкий бюджет трекера (N=24) + медианная фьюжн-склейка кадров",
        verdict: "rejected",
        note: "0 прироста — диагностировано: предел входа (motion blur), не декодер",
      },
    ],
    finding:
      "Реальность: QR-url, DataMatrix-opaque, GS1 DataBar Stacked, Code128 — всё разное. Layout-ROI по физическому гайду — единственное, что реально вскрыло коды. A/B доказал: честный потолок ~9% — это предел качества входа (motion blur стирает модули), лечится мультикадровой агрегацией и условиями съёмки, а не перетюном каскада.",
    shipped:
      "Ансамбль декодеров + layout-ROI каскад (без ML), GTIN-чексумма; дедуп сводит редкие читаемые коды ровно в одну barcode-keyed строку (P0-ключ матчинга).",
  },
  {
    id: "tuning",
    icon: SlidersHorizontal,
    title: "Подбор параметров",
    blurb:
      "Десятки свёрток — измерены, а не угаданы. По одной за раз, оставляем только при измеримом плюсе.",
    stat: "ledger-дисциплина · по 1 свёртке",
    tried: [
      { label: "Детектор: imgsz · conf-операционная точка · optimizer-auto · mosaic/close_mosaic", verdict: "prod" },
      { label: "Трекер: BoT-SORT vs ByteTrack · CMC sparseOptFlow · new_track_thresh=0.82", verdict: "prod" },
      { label: "Дедуп: content-first (равный barcode ⇒ слияние) + wall-clock окно + IoU как сигнал", verdict: "prod" },
      { label: "Лучший кадр: насыщающийся Tenengrad + площадь + det-conf × штраф за край кадра", verdict: "prod" },
      { label: "Кроп: pad-ratio · upscale-таргет · CLAHE; агрегация: min_obs · min_final_conf", verdict: "prod" },
      { label: "Метод: каждый ран — строка ledger, leave-one-video-out, визуальный QA вместо mAP", verdict: "prod" },
      { label: "Перетюн гиперпараметров на малом объёме", verdict: "rejected", note: "инертно, диминишинг" },
    ],
    finding:
      "Правило: одна свёртка за итерацию, оставляем только при измеримом улучшении на held-out. Многие «очевидные» ручки оказались инертны (new_track_thresh залочен 0.82, остальное — шум). Доминируют разрешение, объём данных и качество входа — не тонкая настройка.",
    shipped:
      "Залоченные профили fast / balanced / hq + задокументированные операционные точки + ledger-дисциплина каждого рана.",
  },
];

const VERDICT: Record<
  Verdict,
  { label: string; hint: string; icon: LucideIcon; cls: string }
> = {
  prod: {
    label: "в проде",
    hint: "в финальном пайплайне",
    icon: Check,
    cls: "border-chartwell-blue/40 bg-chartwell-blue/10 text-slate-text",
  },
  explored: {
    label: "исследовали",
    hint: "измерено, не вошло",
    icon: FlaskConical,
    cls: "border-stone-border bg-canvas-fog text-ash-gray",
  },
  rejected: {
    label: "проверили, хуже",
    hint: "отклонено по замеру",
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
          Lenta Tech Life 2026 · Эксперименты
        </p>
        <h1 className="mt-4 font-display text-heading-lg font-medium text-slate-text sm:text-display">
          Что мы перепробовали
        </h1>
        <p className="mx-auto mt-4 max-w-xl text-[16px] leading-[1.6] text-ash-gray">
          Не одна модель из коробки, а измеренный поиск: детекторы, OCR-движки,
          способы читать коды и десятки свёрток. Включая строгие отрицательные
          результаты — где мы доказали замером, что дальше копать бесполезно.
        </p>
        <div className="mt-6 flex flex-wrap items-center justify-center gap-2">
          <Badge variant="accent">
            <Sparkles className="size-3" /> Каждое решение — по замеру, не по наитию
          </Badge>
          <Badge variant="neutral">Полностью локально</Badge>
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
                Открытые датасеты в работе
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
                Что в проде
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
            Каждая ветка — отдельный измеряемый эксперимент: меняем одно,
            смотрим на held-out видео и визуальный QA, оставляем только при
            подтверждённом плюсе. Отрицательные результаты тоже здесь — это и
            есть строгость метода, а не угадайка.
          </p>
        </Card>
      </section>
    </div>
  );
}
