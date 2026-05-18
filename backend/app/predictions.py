"""Reconstruct the non-graded review payload from the graded CSV.

This is the gateway side of the unifying contract (architecture.md §5.6):
the ML service returns only the verbatim 29-column CSV (the graded
artifact) plus non-graded ``meta``. The review screen needs structured
per-tag data (normalized bbox to overlay on the ``<video>``, a seek
fraction, the three field states). So the gateway *parses its own
producer's CSV* back into :class:`JobPredictions` — the same shape the mock
emitted, so the SPA is unchanged.

Coordinate facts (verified against the pipeline):

* The CSV's ``x_min/y_min/x_max/y_max`` are pixels in the **original**
  video frame. The detector rotates frames upright only for the model and
  un-projects boxes back to original coords
  (``detector.unrotate_box_xyxy``). The browser plays that same raw clip,
  so normalised = pixel / raw-frame-size lines up directly — no rotation
  needed here.
* ``frame_timestamp`` is real milliseconds. The SPA seeks
  ``t_frac * video.duration``; with the real raw-clip duration from
  ``meta.video_duration_s`` we set ``t_frac = ms/1000 / duration`` so the
  seek lands exactly on the graded timestamp.

Everything degrades safely: missing ``meta`` (cv2 probe failed) → zeroed
bbox / ``t_frac`` and the review still renders fields + CSV (the graded
download is never affected). ``"нет"`` (absent) and ``""`` (present but
unrecognised) are preserved exactly as the csv module yields them.
"""

from __future__ import annotations

import csv
import io
from uuid import UUID

from app.api.v1.schemas.job import (
    CSV_COLUMNS,
    SUBSTANTIVE_FIELDS,
    BBoxNorm,
    JobPredictions,
    TagPrediction,
)


def _to_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _norm_bbox(fields: dict[str, str], fw: int, fh: int) -> BBoxNorm:
    """Pixel → normalized [0,1], clamped. Zeroed if dims/coords unknown."""
    if fw <= 0 or fh <= 0:
        return BBoxNorm(x1=0.0, y1=0.0, x2=0.0, y2=0.0)
    xs = [_to_int(fields.get("x_min")), _to_int(fields.get("x_max"))]
    ys = [_to_int(fields.get("y_min")), _to_int(fields.get("y_max"))]
    if any(v is None for v in (*xs, *ys)):
        return BBoxNorm(x1=0.0, y1=0.0, x2=0.0, y2=0.0)

    def _c(v: float, hi: int) -> float:
        return round(min(1.0, max(0.0, v / hi)), 4)

    x1, x2 = sorted(xs)  # type: ignore[type-var]
    y1, y2 = sorted(ys)  # type: ignore[type-var]
    return BBoxNorm(x1=_c(x1, fw), y1=_c(y1, fh), x2=_c(x2, fw), y2=_c(y2, fh))


def build_predictions_from_csv(
    *,
    job_id: UUID,
    filename: str,
    csv_text: str,
    meta: dict | None,
    video_url: str,
    csv_url: str,
) -> JobPredictions:
    """Parse the verbatim graded CSV (+ non-graded meta) → JobPredictions."""
    meta = meta or {}
    fw = int(meta.get("frame_width") or 0)
    fh = int(meta.get("frame_height") or 0)
    duration_s = float(meta.get("video_duration_s") or 0.0)

    tags: list[TagPrediction] = []
    reader = csv.reader(io.StringIO(csv_text), lineterminator="\n")
    rows = list(reader)
    # First row is the header; tolerate a degraded/short CSV (e.g. the mock
    # fallback stub) by emitting zero tags rather than raising.
    if rows and rows[0] == CSV_COLUMNS:
        for index, raw in enumerate(rows[1:]):
            if len(raw) != len(CSV_COLUMNS):
                continue
            fields = dict(zip(CSV_COLUMNS, raw))
            ts = _to_int(fields.get("frame_timestamp")) or 0
            t_frac = 0.0
            if duration_s > 0:
                t_frac = round(min(1.0, max(0.0, (ts / 1000.0) / duration_s)), 6)
            tags.append(
                TagPrediction(
                    index=index,
                    color=fields.get("color") or "",
                    frame_timestamp=ts,
                    t_frac=t_frac,
                    bbox=_norm_bbox(fields, fw, fh),
                    fields=fields,
                )
            )

    return JobPredictions(
        job_id=job_id,
        filename=filename,
        columns=CSV_COLUMNS,
        substantive_fields=SUBSTANTIVE_FIELDS,
        video_url=video_url,
        csv_url=csv_url,
        frame_width=fw,
        frame_height=fh,
        tags=tags,
    )
