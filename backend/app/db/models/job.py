"""`jobs` table — one uploaded video → one 29-column CSV result.

One entity = one file (docs/architecture.md §3.4). UUID PK,
``created_at``/``updated_at`` via ``server_default=func.now()``. This
replaces the in-memory mock store that ``routes/jobs.py`` used in the
skeleton; the lifecycle is queued → running → succeeded | failed.

The graded CSV (``result_csv``) is stored verbatim — the gateway serves it
unchanged (it is the contract; never reshape graded columns). The
non-graded review payload (``predictions_json``) is the reconstructed
``JobPredictions`` (see ``app/predictions.py``), persisted once on success
so polling does not re-parse.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    # queued | running | succeeded | failed (schemas.job.JobStatus values).
    status: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    # 0.0 .. 1.0 — mirrors the ML pipeline's single progress signal.
    progress: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    # Original upload name (may be non-ASCII). The bytes live on disk under an
    # ASCII-safe path (``video_path``); this is what the graded CSV's
    # ``filename`` stem must come from — never the on-disk "source.mp4".
    filename: Mapped[str] = mapped_column(String, nullable=False)
    # Absolute path on the shared `uploads` volume, readable by the ML
    # service at the same mount (docs/architecture.md §6).
    video_path: Mapped[str] = mapped_column(String, nullable=False)
    # Detector-only frame pre-rotation chosen in the UI: none | ccw | cw.
    # Passed to the ML service; never alters the stored video or CSV coords.
    rotation: Mapped[str] = mapped_column(
        String(8), nullable=False, default="none", server_default="none"
    )
    # Recognition depth chosen in the UI: "full" | "fast". "fast" makes the
    # ML service cap the heavy Qwen3-VL OCR at the single sharpest crop per
    # tag (vs the config's top-K) — faster, slightly less voting redundancy.
    # Detection/tracking are unchanged. Part of the content-cache key below
    # (a fast run ≠ a full run, so they must not cross-serve).
    mode: Mapped[str] = mapped_column(
        String(8), nullable=False, default="full", server_default="full"
    )
    # sha256 of the uploaded bytes. Together with `rotation` + `mode` it is
    # the content-cache key: a re-upload of the same clip + same rotation +
    # same mode reuses a prior succeeded job's result instead of re-running
    # the pipeline (UI-debug loop without waiting out the full run again).
    content_hash: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )

    # Coarse pipeline stage mirrored from the ML side
    # (detect | finalize | dedup | done). Lets the UI label the bar
    # truthfully — the end-of-video Qwen burst is "finalize", not a guess
    # from the (frame-based, then near-frozen) fraction.
    phase: Mapped[str | None] = mapped_column(String(16), nullable=True)

    rows: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    # The verbatim graded 29-column CSV from the ML service.
    result_csv: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Reconstructed JobPredictions as a JSON string (non-graded review).
    predictions_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
