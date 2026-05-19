// One API module per backend resource (docs/architecture.md §4.3).
//
// These DTOs mirror backend/app/api/v1/schemas/job.py 1:1. The architecture
// (§4.4) wants them *generated* from the gateway's /openapi.json via
// `npm run generate:types` (config: openapi-ts.config.ts) — that needs the
// backend running to fetch the schema. Until that step is run in your env
// they are hand-kept here; swapping to `./generated` is then a one-line
// import change because the shapes are identical.
import { apiClient, apiUrl } from "./client";

export type JobStatus = "queued" | "running" | "succeeded" | "failed";

/** Detector-only frame pre-rotation (never alters stored video / CSV). */
export type Rotation = "none" | "ccw" | "cw";

export interface Job {
  id: string;
  status: JobStatus;
  progress: number; // 0..1
  filename: string;
  rows: number | null;
  error: string | null;
  // Coarse ML stage: detect | finalize | dedup | done (null in mock / before
  // the first poll). Drives the truthful progress label.
  phase: string | null;
  // Place in the single-worker line while queued/running: 0 = running or
  // next up, N>0 = N videos ahead, null = terminal / mock. Videos are
  // processed one at a time (the rented GPU only fits one).
  queue_position: number | null;
  result_csv_url: string | null;
  predictions_url: string | null;
  video_url: string | null;
  created_at: string;
  updated_at: string;
}

export interface BBoxNorm {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface TagPrediction {
  index: number;
  color: string; // white | yellow | green | red
  frame_timestamp: number; // ms — the value written to the CSV
  t_frac: number; // 0..1 — where to seek the uploaded clip for the crop
  bbox: BBoxNorm; // normalized [0,1] for overlaying on the <video>
  fields: Record<string, string>; // 29 columns -> value | "нет" | "" (unrec.)
}

export interface JobPredictions {
  job_id: string;
  filename: string;
  columns: string[]; // canonical 29-column order
  substantive_fields: string[]; // metric-scored subset
  video_url: string;
  csv_url: string;
  frame_width: number;
  frame_height: number;
  tags: TagPrediction[];
}

export const ABSENT = "нет"; // field not present on the tag (task.md §3.3)

export const jobsApi = {
  create: async (video: File, rotation: Rotation = "none"): Promise<Job> => {
    const form = new FormData();
    form.append("video", video);
    // Detector-only pre-rotation. The stored video, the review playback and
    // the graded CSV coords always stay in the uploaded orientation — this
    // only steers how the detector model sees frames (backend → ML).
    form.append("rotation", rotation);
    return (await apiClient.post<Job>("/api/v1/jobs", form)).data;
  },

  list: async (): Promise<Job[]> =>
    (await apiClient.get<Job[]>("/api/v1/jobs")).data,

  get: async (id: string): Promise<Job> =>
    (await apiClient.get<Job>(`/api/v1/jobs/${id}`)).data,

  getPredictions: async (id: string): Promise<JobPredictions> =>
    (await apiClient.get<JobPredictions>(`/api/v1/jobs/${id}/predictions`)).data,

  videoUrl: (id: string): string => apiUrl(`/api/v1/jobs/${id}/video`),
  csvUrl: (id: string): string => apiUrl(`/api/v1/jobs/${id}/result.csv`),
};
