// PLACEHOLDER — one API module per backend resource (docs/architecture.md §4.3).
// DTO types come from src/api/generated/ (npm run generate:types against the
// backend's /openapi.json); hand-written here only until the backend is real.
import { apiClient } from "./client";

export interface Job {
  id: string;
  status: "queued" | "running" | "succeeded" | "failed";
  progress: number;
  filename: string;
  rows: number | null;
  result_csv_url: string | null;
}

export const jobsApi = {
  create: async (video: File): Promise<Job> => {
    const form = new FormData();
    form.append("video", video);
    return (await apiClient.post<Job>("/api/v1/jobs", form)).data;
  },
  get: async (id: string): Promise<Job> =>
    (await apiClient.get<Job>(`/api/v1/jobs/${id}`)).data,
};
