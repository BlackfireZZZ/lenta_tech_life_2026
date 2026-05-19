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
export const DEFAULT_ROTATION: Rotation = "ccw";

/**
 * Recognition depth. "full" runs the heavy text-recognition model on the
 * several sharpest shots of each price tag and votes across them; "fast"
 * uses just the single best shot per tag — markedly quicker, a touch less
 * robust. Detection is identical either way (it is already fast).
 */
export type ProcessingMode = "full" | "fast";

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

/**
 * One raw detector box: `[x1, y1, x2, y2, score]`. The four coords are
 * normalized [0,1] to the original frame (the SAME convention as
 * `TagPrediction.bbox`, so the overlay math is identical); `score` is the
 * detector confidence (always ≥ `DetectorTrace.conf_threshold`).
 */
export type DetectorBox = [number, number, number, number, number];

export interface DetectorFrame {
  t_ms: number; // frame time in ms (exact wall-clock on the real ML path)
  boxes: DetectorBox[];
}

/**
 * Per-frame detector trace — the NON-graded QA artifact behind the
 * "detector markup" view. Lets the reviewer watch the clip with every
 * above-threshold detector box drawn per frame, judging the detector on its
 * own (vs the final one-row-per-tag CSV result). `frames` is time-ordered;
 * a frame with no detection is simply absent (no box is shown there).
 */
export interface DetectorTrace {
  frame_width: number;
  frame_height: number;
  conf_threshold: number; // every box is at/above this detector confidence
  sampled: boolean; // long clips are uniformly strided to bound the payload
  frames: DetectorFrame[];
}

export const ABSENT = "нет"; // field not present on the tag (task.md §3.3)

export const jobsApi = {
  create: async (
    video: File,
    rotation: Rotation = DEFAULT_ROTATION,
    mode: ProcessingMode = "full",
  ): Promise<Job> => {
    const form = new FormData();
    form.append("video", video);
    // Detector-only pre-rotation. The stored video, the review playback and
    // the graded CSV coords always stay in the uploaded orientation — this
    // only steers how the detector model sees frames (backend → ML).
    form.append("rotation", rotation);
    // Recognition depth: "fast" trades OCR voting redundancy for speed
    // (one VLM pass per tag instead of several). Backend → ML.
    form.append("mode", mode);
    return (await apiClient.post<Job>("/api/v1/jobs", form)).data;
  },

  list: async (): Promise<Job[]> =>
    (await apiClient.get<Job[]>("/api/v1/jobs")).data,

  get: async (id: string): Promise<Job> =>
    (await apiClient.get<Job>(`/api/v1/jobs/${id}`)).data,

  getPredictions: async (id: string): Promise<JobPredictions> =>
    (await apiClient.get<JobPredictions>(`/api/v1/jobs/${id}/predictions`)).data,

  // The per-frame detector trace (non-graded). 409 → no trace for this job
  // (older job / ML sent none); the caller falls back to best-frame only.
  getDetections: async (id: string): Promise<DetectorTrace> =>
    (await apiClient.get<DetectorTrace>(`/api/v1/jobs/${id}/detections`)).data,

  videoUrl: (id: string): string => apiUrl(`/api/v1/jobs/${id}/video`),
  csvUrl: (id: string): string => apiUrl(`/api/v1/jobs/${id}/result.csv`),
};
