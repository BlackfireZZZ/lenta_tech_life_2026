"""The backend↔ML contract. This is the ONLY agreed interface between the
gateway and the ML service. The ML service (``../../ml``) implements it; the
gateway calls it through ``client.py``. See docs/architecture.md §5.

Keep this in sync with ``ml/app/contract.py`` — they describe the same JSON.
"""

from pydantic import BaseModel


class ProcessRequest(BaseModel):
    """What the gateway sends the ML service to process one shelf video."""

    video_path: str  # path/URI the ML service can read (shared volume or object store)
    job_id: str  # gateway job id, for correlation/logging


class ProcessResponse(BaseModel):
    """What the ML service returns: the graded CSV plus light metadata.

    ``csv`` is the full 29-column submission text (schema:
    docs/hackathon/task.md). The gateway persists it and serves it verbatim
    — it must never reshape the graded columns.
    """

    csv: str
    rows: int  # number of unique tags / CSV data rows
    meta: dict | None = None  # timings, model versions, etc. — non-graded
