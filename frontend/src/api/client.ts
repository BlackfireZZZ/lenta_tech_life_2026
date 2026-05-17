// The single axios instance. In dev `baseURL` is empty so requests stay
// relative (`/api/...`) and Vite proxies them to the gateway same-origin —
// that keeps the <video> stream and the canvas crop CORS-clean. In a
// deployed build set VITE_API_BASE_URL to the gateway origin.
//
// The full auth client (withCredentials, CSRF + Idempotency request
// interceptors, single-flight 401→refresh) is specified verbatim in
// docs/architecture.md §4.2 and is copied in 1:1 if/when auth lands. This
// flow is anonymous (architecture.md §3.8), so only the instance is needed.
import axios from "axios";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";

export const apiClient = axios.create({
  baseURL: API_BASE,
  withCredentials: true,
});

/** Resolve a gateway-relative path (e.g. a video/CSV URL from the API) to a
 *  URL usable in <video src> / <a href>: as-is in dev (proxied), prefixed
 *  with the API origin in a deployed build. */
export function apiUrl(path: string): string {
  if (/^https?:\/\//.test(path)) return path;
  return `${API_BASE}${path}`;
}
