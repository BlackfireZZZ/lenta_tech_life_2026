// Shared UI helpers. (This module was historically untracked because the
// repo-root .gitignore had a Python `lib/` rule that swallowed
// frontend/src/lib/ — fixed in .gitignore; this is real, tracked source.)
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/** Tailwind-aware className join (clsx + tailwind-merge). */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

/** Humanize a byte count, e.g. 1536 → "1.5 KB". */
export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.min(
    units.length - 1,
    Math.floor(Math.log(bytes) / Math.log(1024)),
  );
  const value = bytes / 1024 ** i;
  return `${value >= 100 || i === 0 ? Math.round(value) : value.toFixed(1)} ${units[i]}`;
}

/**
 * Format a video timecode for display. Input is **milliseconds** from the
 * start of the clip (the graded `frame_timestamp`). Output is `M:SS`
 * (e.g. 12_430 → "0:12"), the raw ms value is shown separately in the UI.
 */
export function formatTimestamp(ms: number): string {
  const total = Math.max(0, Math.floor((ms ?? 0) / 1000));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}
