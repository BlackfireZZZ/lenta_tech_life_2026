import { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  Boxes,
  Camera,
  PackageX,
  ScanSearch,
  Sparkles,
  Tags,
} from "lucide-react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Spinner } from "@/components/ui/spinner";
import { cn } from "@/lib/utils";
import {
  cropUrl,
  loadAudit,
  loadVideoIndex,
  type ProductCard,
  type ShelfAlert,
  type ShelfAudit,
  type VideoIndexItem,
} from "@/api/shelfAudit";

// ─────────────────────────────────────────────────────────────────────────
// Killer feature, beside the graded CSV: from one robot pass we surface
// out-of-stock shelves and price-law violations (товар без ценника), plus a
// product card per item. Fully automatic. Reads a static fixture built by
// scripts/make_shelf_audit_fixture.py — see docs/shelf-audit.md.
// ─────────────────────────────────────────────────────────────────────────

function fmtTime(s: number | null): string {
  if (s == null || !isFinite(s)) return "—";
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

function Stat({
  value,
  label,
  tone,
}: {
  value: number | string;
  label: string;
  tone: "danger" | "warning" | "accent" | "neutral";
}) {
  const ring = {
    danger: "text-red-600",
    warning: "text-amber-600",
    accent: "text-chartwell-blue",
    neutral: "text-slate-text",
  }[tone];
  return (
    <Card className="flex flex-col gap-1 p-5">
      <span className={cn("font-display text-display font-medium tabular-nums", ring)}>
        {value}
      </span>
      <span className="text-caption text-ash-gray">{label}</span>
    </Card>
  );
}

function AlertRow({ alert, videoId }: { alert: ShelfAlert; videoId: string }) {
  const oos = alert.type === "OUT_OF_STOCK";
  const img = cropUrl(videoId, alert.evidence_crop);
  return (
    <Card className="flex gap-4 p-4">
      <div className="size-20 shrink-0 overflow-hidden rounded-input border border-stone-border bg-canvas-fog">
        {img ? (
          <img src={img} alt="" className="size-full object-cover" loading="lazy" />
        ) : (
          <div className="grid size-full place-items-center text-steel-gray">
            <Camera className="size-5" />
          </div>
        )}
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          {oos ? (
            <Badge variant="danger">
              <PackageX className="size-3" /> Пустая полка
            </Badge>
          ) : (
            <Badge variant="warning">
              <Tags className="size-3" /> Товар без ценника
            </Badge>
          )}
          <span className="font-mono text-[11px] text-steel-gray">{alert.id}</span>
        </div>
        <p className="mt-2 text-[13px] leading-[1.5] text-slate-text">
          {oos
            ? "Ценник распознан, но товара над ним нет — упущенные продажи."
            : "Товар на полке без ценника — нарушение (ЗоЗПП требует цену на каждом товаре)."}
        </p>
        <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[12px] text-ash-gray">
          <span>⏱ {fmtTime(alert.timestamp_s)}</span>
          <span>виден {alert.persistence_frames} кадров</span>
          {alert.product?.facing_count != null && (
            <span>{alert.product.facing_count} фейсингов</span>
          )}
          {alert.track_id != null && (
            <span className="font-mono">трек {alert.track_id}</span>
          )}
        </div>
      </div>
    </Card>
  );
}

function CardTile({ card, videoId }: { card: ProductCard; videoId: string }) {
  const img = cropUrl(videoId, card.best_crop);
  return (
    <Card className="flex flex-col overflow-hidden">
      <div className="relative aspect-[3/4] bg-canvas-fog">
        {img ? (
          <img src={img} alt="" className="size-full object-cover" loading="lazy" />
        ) : (
          <div className="grid size-full place-items-center text-steel-gray">
            <Camera className="size-6" />
          </div>
        )}
        <span className="absolute right-2 top-2">
          <Badge variant="accent">×{card.facing_count} фейсингов</Badge>
        </span>
      </div>
      <div className="flex flex-col gap-1 p-3">
        <p className="truncate text-[13px] font-medium text-slate-text">
          {card.name ?? (
            <span className="text-steel-gray">Название — после интеграции OCR</span>
          )}
        </p>
        <div className="flex items-center justify-between text-[12px]">
          <span className={card.price != null ? "text-slate-text" : "text-steel-gray"}>
            {card.price != null ? `${card.price} ₽` : "цена — позже"}
          </span>
          <span className="font-mono text-[11px] text-ash-gray">
            {fmtTime(card.seen_from_s)}–{fmtTime(card.seen_to_s)}
          </span>
        </div>
        <span className="truncate font-mono text-[11px] text-steel-gray">
          {card.barcode ?? "штрихкод — после интеграции OCR"}
        </span>
      </div>
    </Card>
  );
}

export default function ShelfAuditPage() {
  const [videos, setVideos] = useState<VideoIndexItem[]>([]);
  const [videoId, setVideoId] = useState<string | null>(null);
  const [audit, setAudit] = useState<ShelfAudit | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadVideoIndex()
      .then((idx) => {
        setVideos(idx.videos);
        setVideoId(idx.videos[0]?.id ?? null);
        if (!idx.videos.length) {
          setError("Фикстура пуста — запустите run_shelf_audit.py и make_shelf_audit_fixture.py.");
          setLoading(false);
        }
      })
      .catch((e) => {
        setError(
          "Нет данных аудита. Сгенерируйте фикстуру: run_shelf_audit.py → make_shelf_audit_fixture.py. " +
            String(e),
        );
        setLoading(false);
      });
  }, []);

  useEffect(() => {
    if (!videoId) return;
    setLoading(true);
    loadAudit(videoId)
      .then((a) => {
        setAudit(a);
        setError(null);
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, [videoId]);

  const s = audit?.summary;
  const alerts = useMemo(
    () =>
      [...(audit?.alerts ?? [])].sort((a, b) =>
        a.type === b.type ? a.timestamp_s - b.timestamp_s : a.type < b.type ? -1 : 1,
      ),
    [audit],
  );

  return (
    <div className="flex flex-col gap-10">
      {/* Hero */}
      <section className="mx-auto max-w-2xl pt-6 text-center">
        <p className="text-caption font-medium uppercase tracking-[0.14em] text-chartwell-blue">
          Lenta Tech Life 2026 · Доп. фича
        </p>
        <h1 className="mt-4 font-display text-heading-lg font-medium text-slate-text sm:text-display">
          Аудит полки
        </h1>
        <p className="mx-auto mt-4 max-w-xl text-[16px] leading-[1.6] text-ash-gray">
          Тот же проезд робота, что даёт зачётный CSV, попутно находит{" "}
          <b className="text-slate-text">пустые полки</b> и{" "}
          <b className="text-slate-text">товары без ценника</b>, и собирает
          карточку на каждый товар. Полностью автоматически, рядом с зачётным
          CSV — не вместо него.
        </p>
        <div className="mt-6 flex flex-wrap items-center justify-center gap-2">
          <Badge variant="accent">
            <Sparkles className="size-3" /> Полностью автоматически
          </Badge>
          <Badge variant="info">Рядом с зачётным CSV</Badge>
          <Badge variant="neutral">Локально · без облака</Badge>
        </div>
      </section>

      {videos.length > 1 && (
        <div className="mx-auto flex flex-wrap justify-center gap-2">
          {videos.map((v) => (
            <button
              key={v.id}
              type="button"
              onClick={() => setVideoId(v.id)}
              className={cn(
                "rounded-pill border px-3 py-1.5 text-caption font-medium transition-colors",
                v.id === videoId
                  ? "border-chartwell-blue/40 bg-sky-tint/40 text-slate-text"
                  : "border-stone-border text-ash-gray hover:text-slate-text",
              )}
            >
              {v.id}
            </button>
          ))}
        </div>
      )}

      {loading && (
        <div className="grid place-items-center py-24 text-ash-gray">
          <Spinner className="size-6" />
        </div>
      )}

      {error && !loading && (
        <Card className="mx-auto max-w-xl p-6 text-center">
          <AlertTriangle className="mx-auto size-6 text-amber-500" />
          <p className="mt-3 text-[14px] leading-[1.6] text-ash-gray">{error}</p>
        </Card>
      )}

      {audit && s && !loading && !error && (
        <>
          {/* Stats */}
          <section className="mx-auto grid w-full max-w-4xl grid-cols-2 gap-3 sm:grid-cols-4">
            <Stat value={s.n_out_of_stock} label="Пустые полки" tone="danger" />
            <Stat value={s.n_missing_price_tag} label="Товары без ценника" tone="warning" />
            <Stat value={s.n_product_cards} label="Распознано товаров" tone="accent" />
            <Stat value={s.n_relations_ok} label="Ценник ↔ товар" tone="neutral" />
          </section>
          {s.missing_tag_facing_level_would_be != null && (
            <p className="mx-auto -mt-6 max-w-3xl text-center text-[12px] text-steel-gray">
              Группировка фейсингов в товар убирает ложные срабатывания:{" "}
              {s.n_product_tracks} фейсингов → {s.n_product_cards} карточек;
              «товар без ценника» {s.missing_tag_facing_level_would_be} →{" "}
              {s.n_missing_price_tag} на уровне товара.
            </p>
          )}

          <section className="mx-auto grid w-full max-w-6xl gap-8 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
            {/* Alerts feed */}
            <div className="flex flex-col gap-3">
              <div className="flex items-center gap-2">
                <ScanSearch className="size-5 text-chartwell-blue" />
                <h2 className="font-display text-heading-sm font-medium text-slate-text">
                  Алерты
                </h2>
                <Badge variant="neutral">{alerts.length}</Badge>
              </div>
              {alerts.length === 0 ? (
                <Card className="p-6 text-center text-[14px] text-ash-gray">
                  Нарушений не найдено на этом видео.
                </Card>
              ) : (
                alerts.map((a) => (
                  <AlertRow key={a.id} alert={a} videoId={audit.video_id} />
                ))
              )}
            </div>

            {/* Product cards */}
            <div className="flex flex-col gap-3">
              <div className="flex items-center gap-2">
                <Boxes className="size-5 text-chartwell-blue" />
                <h2 className="font-display text-heading-sm font-medium text-slate-text">
                  Карточки товаров
                </h2>
                <Badge variant="neutral">{audit.cards.length}</Badge>
              </div>
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
                {audit.cards.map((c) => (
                  <CardTile key={c.card_id} card={c} videoId={audit.video_id} />
                ))}
              </div>
            </div>
          </section>

          <section className="mx-auto w-full max-w-3xl">
            <Card className="flex flex-col items-center gap-2 p-6 text-center">
              <p className="text-caption font-medium uppercase tracking-[0.12em] text-chartwell-blue">
                Зачем это
              </p>
              <p className="max-w-xl text-[14px] leading-[1.65] text-ash-gray">
                Робот, который и так читает ценники, тем же проездом превращается
                в аудитора полки: видит упущенные продажи и нарушения
                выкладки. Цена, название и штрихкод в карточках подставятся при
                интеграции с основным распознаванием.
              </p>
            </Card>
          </section>
        </>
      )}
    </div>
  );
}
