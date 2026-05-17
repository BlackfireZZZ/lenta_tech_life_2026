// PLACEHOLDER. The real axios client (withCredentials, CSRF + Idempotency
// request interceptors, single-flight 401→refresh) is specified verbatim in
// docs/architecture.md §4.2 and is copied in 1:1 when auth lands.
import axios from "axios";

export const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000",
  withCredentials: true,
});
