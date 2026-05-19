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
    # Recognition depth: "full" | "fast". "full" (default) runs the heavy
    # Qwen3-VL OCR on the top-K sharpest crops per tag; "fast" caps it at the
    # single sharpest crop (~K× fewer VLM calls). Detection/tracking are
    # unchanged (cheap); only OCR redundancy is traded for speed. Unknown
    # value → "full". See the backend mirror for the full rationale.
    mode: str = "full"


class ProcessResponse(BaseModel):
    csv: str  # full 29-column submission text (schema: docs/hackathon/task.md)
    rows: int  # number of unique tags / CSV data rows
    meta: dict | None = None  # timings, model versions — non-graded
    # Per-frame detector trace — NON-graded QA side-artifact (additive,
    # default None so older callers are unaffected; the graded `csv` is
    # never derived from it). Shape (price_tag_pipeline.frame_trace):
    #   {frame_width, frame_height, conf_threshold, sampled,
    #    frames: [{t_ms, boxes: [[x1,y1,x2,y2,score], ...]}]}
    # boxes are normalised [0,1] to the original frame (same convention as
    # the best-frame bbox). Lets the UI replay the clip with the raw
    # detector output overlaid. See the backend mirror.
    detections: dict | None = None
