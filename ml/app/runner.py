"""Bridge from the HTTP contract to the real price-tag pipeline.

This is the ONLY seam between the deployable service and the research
codebase. The model lives in the ``price_tag_pipeline`` package
(``projects/price_tag_pipeline/``); here we call its public entry point
(``PriceTagPipeline(cfg).run``) and render the graded CSV with
``submission.final_tags_to_csv``. Keeping the seam this thin means
training/experiment churn there never breaks the service contract.

Real vs. mock (so the monorepo still boots without GPU/weights — the
architecture.md §5 principle):

* ``ML_MOCK=1``                  → always the fake CSV.
* pipeline import fails          → fall back to the fake CSV, logged once.
* otherwise                      → the real pipeline runs.

Config: ``ML_PIPELINE_CONFIG`` (default ``configs/balanced.yaml``).
Progress is streamed into ``progress_registry`` keyed by ``job_id`` for the
additive ``GET /progress/{job_id}`` endpoint — the ``ProcessRequest`` /
``ProcessResponse`` contract itself is unchanged.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

from app.contract import ProcessRequest, ProcessResponse
from app import progress_registry

LOGGER = logging.getLogger("ml.runner")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PIPELINE_DIR = _REPO_ROOT / "projects" / "price_tag_pipeline"
_PIPELINE_SRC = _PIPELINE_DIR / "src"
if str(_PIPELINE_SRC) not in sys.path:
    # Import the package without installing it — same convention the
    # scripts/ entry points use. Heavy deps (ultralytics, ...) only load
    # when run() actually touches the detector, so import stays cheap.
    sys.path.insert(0, str(_PIPELINE_SRC))

_MOCK_CSV = "filename,frame_timestamp,x1,y1,x2,y2,barcode,name,price,...\n"

# Cached: did the pipeline import succeed? (None = not yet probed.)
_real_available: bool | None = None


def _mock_forced() -> bool:
    return os.getenv("ML_MOCK", "").strip().lower() in {"1", "true", "yes", "on"}


def _config_path() -> Path:
    env = os.getenv("ML_PIPELINE_CONFIG", "").strip()
    if env:
        p = Path(env).expanduser()
        return p if p.is_absolute() else (_REPO_ROOT / p)
    return _PIPELINE_DIR / "configs" / "balanced.yaml"


def current_mode() -> str:
    """'mock' or 'real' — what the next /process call will do. For /health."""
    if _mock_forced():
        return "mock"
    return "real" if _probe_real() else "mock"


def _probe_real() -> bool:
    """True if the pipeline package imports. Cached; never raises."""
    global _real_available
    if _real_available is None:
        try:
            import price_tag_pipeline.pipeline  # noqa: F401
            import price_tag_pipeline.submission  # noqa: F401

            _real_available = True
        except Exception as exc:  # missing heavy deps → stay mock, but boot
            LOGGER.warning("Pipeline import failed; serving MOCK CSV: %s", exc)
            _real_available = False
    return _real_available


def run_pipeline(req: ProcessRequest) -> ProcessResponse:
    if _mock_forced() or not _probe_real():
        progress_registry.record(req.job_id, {
            "phase": "done", "fraction": 1.0, "message": "mock", "mock": True,
        })
        return ProcessResponse(csv=_MOCK_CSV, rows=0,
                               meta={"mock": True, "job_id": req.job_id})

    from price_tag_pipeline.config import load_config
    from price_tag_pipeline.pipeline import PriceTagPipeline
    from price_tag_pipeline.progress import ProgressEvent
    from price_tag_pipeline.submission import final_tags_to_csv

    def _on_progress(ev: ProgressEvent) -> None:
        progress_registry.record(req.job_id, ev.as_dict())

    cfg_path = _config_path()
    started = time.time()
    LOGGER.info("job=%s start video=%s config=%s", req.job_id, req.video_path, cfg_path)

    cfg = load_config(cfg_path)
    tags = PriceTagPipeline(cfg).run(req.video_path, progress=_on_progress)

    # task.md §3.4: the released Lenta CSVs use a bare stem as `filename`.
    filename = Path(req.video_path).stem
    csv_text = final_tags_to_csv(tags, filename=filename)
    elapsed = round(time.time() - started, 2)
    LOGGER.info("job=%s done rows=%d elapsed=%ss", req.job_id, len(tags), elapsed)

    return ProcessResponse(
        csv=csv_text,
        rows=len(tags),
        meta={
            "job_id": req.job_id,
            "mock": False,
            "config": cfg_path.name,
            "elapsed_s": elapsed,
        },
    )
