"""Pydantic DTOs for the `jobs` resource — the gateway's public contract.

A *job* = one uploaded shelf video → one 29-column result CSV. This is the
only business resource in the price-tag product (there is no generic CRUD).
See docs/architecture.md §3.5.

The CSV is the **graded** artifact and is served verbatim
(``GET /jobs/{id}/result.csv``). ``JobPredictions`` is a *non-graded*
review/visualization layer (task.md §13 "bbox visualization") that lets the
UI show the source video, the predicted timestamp, the price-tag crop and the
recognized fields. Both are built from the same per-tag data so they never
disagree.
"""

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, ConfigDict

# --- The canonical 29-column CSV schema (docs/hackathon/task.md §3) ---------
# 18 fields from the tag itself + 11 from the QR code. Order is the contract:
# the CSV header and every data row follow this exact sequence. Technical
# fields (filename, frame_timestamp, the bbox) are used only for GT matching
# and are NOT scored — see task.md §5.2.
CSV_COLUMNS: list[str] = [
    # --- from the tag itself (18) ---
    "filename",
    "product_name",
    "price_default",
    "price_card",
    "price_discount",
    "barcode",
    "discount_amount",
    "id_sku",
    "print_datetime",
    "code",
    "additional_info",
    "color",
    "special_symbols",
    "frame_timestamp",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    # --- from the QR code (11) ---
    "qr_code_barcode",
    "price1_qr",
    "price2_qr",
    "price3_qr",
    "price4_qr",
    "wholesale_level_1_count",
    "wholesale_level_1_price",
    "wholesale_level_2_count",
    "wholesale_level_2_price",
    "action_price_qr",
    "action_code_qr",
]

# Substantive fields are everything except the technical ones. Only these
# count toward the metric; the UI uses this split to score per-tag
# completeness honestly (task.md §5.2).
TECHNICAL_FIELDS: set[str] = {
    "filename",
    "frame_timestamp",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
}
SUBSTANTIVE_FIELDS: list[str] = [c for c in CSV_COLUMNS if c not in TECHNICAL_FIELDS]


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class JobResponse(BaseModel):
    """Returned by POST /jobs and GET /jobs/{id}."""

    id: UUID
    status: JobStatus
    progress: float = 0.0  # 0.0 .. 1.0
    filename: str
    rows: int | None = None  # unique tags found (set when succeeded)
    error: str | None = None
    # Coarse ML stage (detect | finalize | dedup | done); None in MOCK_MODE
    # and before the first poll. The UI labels the bar from this.
    phase: str | None = None
    # All three are set together when the job succeeds. URLs are relative to
    # the gateway origin so the SPA can serve them through its dev proxy.
    result_csv_url: str | None = None
    predictions_url: str | None = None
    video_url: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class BBoxNorm(BaseModel):
    """Bounding box in **normalized** [0, 1] coordinates relative to the
    displayed frame, so the SPA can overlay it on a ``<video>`` of any size
    without knowing the source resolution. Pixel coordinates for the CSV are
    derived from these against the nominal frame size in ``JobPredictions``.
    """

    x1: float
    y1: float
    x2: float
    y2: float


class TagPrediction(BaseModel):
    """One unique price tag = one CSV row, plus what the UI needs to show it.

    ``fields`` holds every one of the 29 columns as the *string it would be
    written as in the CSV*, so the three task.md §3.3 states are visible
    end-to-end:

    - a real value           → the recognized text
    - ``"нет"``              → the field is absent on this tag
    - ``""`` (empty string)  → present on the tag but not recognized

    The frontend renders these three states distinctly — that distinction is
    worth points (task.md §5.3) and is the whole reason the review screen
    exists, instead of a bare video→CSV.
    """

    index: int
    color: str  # white | yellow | green | red (also mirrored in `fields`)
    frame_timestamp: int  # ms from video start — the value written to the CSV
    # Where to seek the *uploaded* clip for the crop. The mock cannot know the
    # real clip duration, so it emits a fraction and the SPA multiplies by the
    # actual <video>.duration. `frame_timestamp` stays the graded ms value.
    t_frac: float
    bbox: BBoxNorm
    fields: dict[str, str]  # all 29 CSV_COLUMNS -> string (value / "нет" / "")


class JobPredictions(BaseModel):
    """Non-graded review payload for one finished job (GET
    /jobs/{id}/predictions). The graded CSV is served separately and verbatim.
    """

    job_id: UUID
    filename: str
    columns: list[str]  # == CSV_COLUMNS, the canonical order
    substantive_fields: list[str]  # the metric-scored subset
    video_url: str
    csv_url: str
    frame_width: int  # nominal frame size the pixel bbox / CSV coords assume
    frame_height: int
    tags: list[TagPrediction]
