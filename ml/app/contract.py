"""ML-side mirror of the backend↔ML contract.

MUST stay byte-compatible with ``backend/app/ml/schemas.py`` — same JSON
shape, two owners. See docs/architecture.md §5.
"""

from pydantic import BaseModel


class ProcessRequest(BaseModel):
    video_path: str  # path/URI this service can read
    job_id: str  # gateway job id, for correlation/logging
    # Original upload name; its bare stem is the graded CSV `filename` cell.
    # "" → fall back to the video_path stem (backward compat). See the
    # backend mirror for why this is required for correct GT matching.
    filename: str = ""
    # Detector-only frame pre-rotation: none | ccw | cw. Steers
    # cfg.detector.frame_rotation (how the model sees frames). The stored
    # video, the review playback and the graded CSV coords are ALWAYS the
    # original orientation — boxes are un-projected back, this never rotates
    # any output. Default "none": an uploaded clip is trusted to be in its
    # real-life orientation and the detector sees it untouched (verified: a
    # normal upright phone clip gets clean boxes at "none", garbage at "ccw").
    # The UI rotate button is the per-clip override for sideways footage.
    rotation: str = "none"


class ProcessResponse(BaseModel):
    csv: str  # full 29-column submission text (schema: docs/hackathon/task.md)
    rows: int  # number of unique tags / CSV data rows
    meta: dict | None = None  # timings, model versions — non-graded
