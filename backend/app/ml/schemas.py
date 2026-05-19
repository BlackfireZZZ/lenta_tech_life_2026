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
    # Original upload name. The graded CSV's `filename` cell is its bare stem
    # (released Lenta CSVs use e.g. `25_2-10`). REQUIRED for correct GT
    # matching: the bytes are stored on disk under an ASCII-safe
    # "source.<ext>", so the ML service cannot recover the real name from
    # `video_path`. Optional/"" only for backward compat (then the ML side
    # falls back to the video_path stem).
    filename: str = ""
    # Detector-only frame pre-rotation: none | ccw | cw. Steers
    # cfg.detector.frame_rotation (how the model sees frames). The stored
    # video, the review playback and the graded CSV coords are ALWAYS the
    # original orientation (boxes un-projected back). Default "none": an
    # uploaded clip is trusted to be in its real-life orientation and the
    # detector sees it untouched. The UI rotate button is the per-clip
    # override for sideways footage (e.g. robot cam mounted 90° CW).
    rotation: str = "none"
    # Recognition depth: "full" | "fast". "full" (default = canonical
    # balanced.yaml) runs the heavy Qwen3-VL OCR on the top-K sharpest crops
    # per tag; "fast" caps that at the single sharpest crop (~K× fewer VLM
    # calls — the slow part). Detection/tracking are unchanged either way
    # (they are cheap); only OCR redundancy is traded for speed. Unknown
    # value → "full" (no override; keep the config's behaviour).
    mode: str = "full"


class ProcessResponse(BaseModel):
    """What the ML service returns: the graded CSV plus light metadata.

    ``csv`` is the full 29-column submission text (schema:
    docs/hackathon/task.md). The gateway persists it and serves it verbatim
    — it must never reshape the graded columns.
    """

    csv: str
    rows: int  # number of unique tags / CSV data rows
    meta: dict | None = None  # timings, model versions, etc. — non-graded
