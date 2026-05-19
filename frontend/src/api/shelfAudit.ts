// Shelf-audit data access. The backend/ml is a mocked skeleton (docs/index.md),
// so the page reads a STATIC fixture served from Vite's public/ — zero-backend
// jury demo. Built by scripts/make_shelf_audit_fixture.py from
// outputs/shelf_audit/. Contract mirrors shelf_analytics/schema.py.

const FIXTURE_BASE = "/shelf-audit";

export type AlertType = "OUT_OF_STOCK" | "MISSING_PRICE_TAG";

export interface ShelfAlert {
  id: string;
  type: AlertType;
  severity: string;
  video_id: string;
  timestamp_s: number;
  frame_idx: number;
  bbox_xyxy: number[];
  track_id: number | null;
  evidence_crop: string | null;
  price_tag: Record<string, unknown> | null;
  product: { card_id?: string; facing_count?: number; member_track_ids?: number[] } | null;
  first_seen_s: number | null;
  last_seen_s: number | null;
  persistence_frames: number;
}

export interface ProductCard {
  card_id: string;
  product_track_id: number;
  best_crop: string | null;
  facing_count: number;
  seen_from_s: number | null;
  seen_to_s: number | null;
  matched_price_tag_track_id: number | null;
  price: number | null;
  loyalty_price: number | null;
  name: string | null;
  barcode: string | null;
  catalog: Record<string, unknown> | null;
  embedding_dim: number | null;
}

export interface ShelfAuditSummary {
  n_out_of_stock: number;
  n_missing_price_tag: number;
  n_product_cards: number;
  n_relations_ok: number;
  n_relations_ambiguous: number;
  n_product_tracks: number;
  n_price_tag_tracks: number;
  missing_tag_facing_level_would_be?: number;
  est_lost_revenue_rub: number | null;
}

export interface ShelfAudit {
  video_id: string;
  fps: number;
  summary: ShelfAuditSummary;
  relations: unknown[];
  alerts: ShelfAlert[];
  cards: ProductCard[];
  metadata: Record<string, unknown>;
}

export interface VideoIndexItem {
  id: string;
  n_alerts: number;
  n_cards: number;
  n_out_of_stock: number;
  n_missing_price_tag: number;
}

async function getJson<T>(url: string): Promise<T> {
  const r = await fetch(url, { cache: "no-cache" });
  if (!r.ok) throw new Error(`${r.status} ${url}`);
  return (await r.json()) as T;
}

export function loadVideoIndex(): Promise<{ videos: VideoIndexItem[] }> {
  return getJson(`${FIXTURE_BASE}/index.json`);
}

export function loadAudit(videoId: string): Promise<ShelfAudit> {
  return getJson(`${FIXTURE_BASE}/${videoId}/audit.json`);
}

/** Resolve a crop path ("cards/x.jpg" | "crops/x.jpg") to a fixture URL. */
export function cropUrl(videoId: string, rel: string | null): string | null {
  return rel ? `${FIXTURE_BASE}/${videoId}/${rel}` : null;
}
