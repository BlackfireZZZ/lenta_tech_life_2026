"""Pydantic DTOs for the `jobs` resource — the gateway's public contract.

A *job* = one uploaded shelf video → one 29-column result CSV. This is the
only business resource in the price-tag product (there is no generic CRUD).
See docs/architecture.md §3.5.
"""

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, ConfigDict


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
    result_csv_url: str | None = None  # set when succeeded
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
