"""ML-side mirror of the backend↔ML contract.

MUST stay byte-compatible with ``backend/app/ml/schemas.py`` — same JSON
shape, two owners. See docs/architecture.md §5.
"""

from pydantic import BaseModel


class ProcessRequest(BaseModel):
    video_path: str  # path/URI this service can read
    job_id: str  # gateway job id, for correlation/logging


class ProcessResponse(BaseModel):
    csv: str  # full 29-column submission text (schema: docs/hackathon/task.md)
    rows: int  # number of unique tags / CSV data rows
    meta: dict | None = None  # timings, model versions — non-graded
