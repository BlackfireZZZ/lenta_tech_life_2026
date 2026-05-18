"""`jobs` resource — upload a shelf video, poll status, review + download.

    POST /api/v1/jobs                  multipart video -> {id, status: queued}
    GET  /api/v1/jobs/{id}             -> {id, status, progress, *_url?}
    GET  /api/v1/jobs/{id}/video       -> the uploaded clip (range-enabled)
    GET  /api/v1/jobs/{id}/predictions -> non-graded per-tag review JSON
    GET  /api/v1/jobs/{id}/result.csv  -> the graded 29-column CSV (verbatim)

Two modes, switched by ``settings.MOCK_MODE`` (architecture.md §3.2 — flip
off layer-by-layer):

* ``MOCK_MODE=true``  — the standalone skeleton: an in-memory store fakes a
  job that "progresses" on each poll, tags fabricated deterministically per
  job id (``app/jobs_mock.py``). No DB, no ML. Kept so the frontend e2e and
  reviews still work without infra.
* ``MOCK_MODE=false`` — the real product (docker-compose): jobs persist in
  Postgres; processing runs **out-of-band** (architecture.md §5.3) — the
  request returns immediately and an asyncio task calls the ML service,
  polling its ``/progress`` to drive ``JobResponse.progress`` while the
  (minutes-long) pipeline runs, then persists the verbatim graded CSV and
  the reconstructed review payload.

The graded CSV is served byte-for-byte as the ML service produced it — the
gateway never reshapes graded columns.
"""

from __future__ import annotations

import asyncio
import shutil
from datetime import datetime, timezone
from pathlib import Path
from tempfile import gettempdir
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, PlainTextResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.schemas.job import JobPredictions, JobResponse, JobStatus
from app.cache.redis import get_redis_client
from app.core.config import settings
from app.core.logger import logger
from app.db.models.job import Job as _JobModel
from app.db.session import session_scope
from app.jobs_mock import build_csv, build_predictions, generate_tags
from app.ml.client import ml_client
from app.ml.schemas import ProcessRequest
from app.predictions import build_predictions_from_csv

router = APIRouter(prefix="/jobs", tags=["Jobs"])


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Shared: where uploaded clips live
# ---------------------------------------------------------------------------
# In docker-compose the `uploads` volume is mounted at the same path in both
# `backend` and `ml`, so the absolute path the gateway writes is exactly the
# path the ML service reads (architecture.md §6). Falls back to a temp dir
# when that root is not writable (local dev without the volume).


def _upload_dir(job_id: UUID) -> Path:
    root = Path(settings.UPLOAD_ROOT)
    try:
        d = root / str(job_id)
        d.mkdir(parents=True, exist_ok=True)
        return d
    except OSError:
        d = Path(gettempdir()) / "lenta_jobs" / str(job_id)
        d.mkdir(parents=True, exist_ok=True)
        return d


def _save_upload(video: UploadFile, job_id: UUID) -> tuple[str, Path]:
    """Persist the clip under an ASCII-safe name; return (orig_name, path).

    The original name may be non-ASCII; keep it only as metadata and store
    the bytes as ``source.<ext>`` so non-ASCII paths never break I/O.
    """
    filename = video.filename or "upload.mp4"
    ext = Path(filename).suffix.lower() or ".mp4"
    video_path = _upload_dir(job_id) / f"source{ext}"
    with video_path.open("wb") as out:
        shutil.copyfileobj(video.file, out)
    return filename, video_path


# ===========================================================================
# MOCK_MODE path — in-memory, no DB / no ML (standalone skeleton)
# ===========================================================================

_MOCK_JOBS: dict[UUID, dict] = {}


def _mock_get_or_404(job_id: UUID) -> dict:
    job = _MOCK_JOBS.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")
    return job


def _mock_tags_for(job: dict) -> list:
    if job["tags"] is None:
        job["tags"] = generate_tags(job["id"], job["filename"])
    return job["tags"]


def _mock_create(video: UploadFile) -> JobResponse:
    job_id = uuid4()
    filename, video_path = _save_upload(video, job_id)
    _MOCK_JOBS[job_id] = {
        "id": job_id,
        "status": JobStatus.queued,
        "progress": 0.0,
        "filename": filename,
        "video_path": video_path,
        "tags": None,
        "rows": None,
        "error": None,
        "result_csv_url": None,
        "predictions_url": None,
        "video_url": None,
        "created_at": _now(),
        "updated_at": _now(),
    }
    return JobResponse.model_validate(_MOCK_JOBS[job_id])


def _mock_get(job_id: UUID) -> JobResponse:
    job = _mock_get_or_404(job_id)
    if job["status"] in (JobStatus.queued, JobStatus.running):
        job["status"] = JobStatus.running
        job["progress"] = round(min(job["progress"] + 0.34, 1.0), 2)
        if job["progress"] >= 1.0:
            job["status"] = JobStatus.succeeded
            job["rows"] = len(_mock_tags_for(job))
            base = f"/api/v1/jobs/{job_id}"
            job["result_csv_url"] = f"{base}/result.csv"
            job["predictions_url"] = f"{base}/predictions"
            job["video_url"] = f"{base}/video"
        job["updated_at"] = _now()
    return JobResponse.model_validate(job)


# ===========================================================================
# Real path — Postgres + out-of-band ML processing
# ===========================================================================

# Keep strong refs to background tasks so they are not GC'd mid-run.
_BG_TASKS: set[asyncio.Task] = set()


def _job_response(job) -> JobResponse:
    base = f"/api/v1/jobs/{job.id}"
    done = job.status == JobStatus.succeeded.value
    return JobResponse(
        id=job.id,
        status=JobStatus(job.status),
        progress=job.progress,
        filename=job.filename,
        rows=job.rows,
        error=job.error,
        result_csv_url=f"{base}/result.csv" if done else None,
        predictions_url=f"{base}/predictions" if done else None,
        video_url=f"{base}/video" if done else None,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


async def _stop(task: asyncio.Task) -> None:
    """Cancel a task and wait for it to actually unwind (idempotent)."""
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):  # noqa: BLE001
        pass


async def _poll_progress(job_id: UUID) -> None:
    """Mirror the ML pipeline's progress into the job row until cancelled.

    Best-effort: a flaky poll never fails the job (the outcome is decided by
    ``ml_client.process``). Only advances; never marks the job done.
    """
    while True:
        await asyncio.sleep(1.5)
        snap = await ml_client.get_progress(str(job_id))
        frac = snap.get("fraction")
        if frac is None:
            continue
        try:
            value = max(0.0, min(0.99, float(frac)))
        except (TypeError, ValueError):
            continue
        try:
            async with session_scope() as s:
                job = await s.get(_JobModel, job_id)
                if job is None or job.status != JobStatus.running.value:
                    return
                if value > job.progress:
                    job.progress = value
        except Exception as exc:  # pragma: no cover - never fatal
            logger.debug("progress write failed (job=%s): %s", job_id, exc)


async def _process_job(job_id: UUID, video_path: str, filename: str) -> None:
    """Out-of-band worker: run the ML pipeline, persist its result.

    Owns its DB sessions (the request-scoped one is long gone by now).
    Any failure → the job is marked ``failed`` with the message; it never
    crashes the server.
    """
    async with session_scope() as s:
        job = await s.get(_JobModel, job_id)
        if job is None:
            return
        job.status = JobStatus.running.value
        job.progress = 0.0

    poller = asyncio.create_task(_poll_progress(job_id))
    try:
        resp = await ml_client.process(
            ProcessRequest(
                video_path=video_path, job_id=str(job_id), filename=filename
            )
        )
    except Exception as exc:
        await _stop(poller)
        logger.exception("job=%s ML processing failed", job_id)
        async with session_scope() as s:
            job = await s.get(_JobModel, job_id)
            if job is not None:
                job.status = JobStatus.failed.value
                job.error = str(exc)[:2000]
                job.progress = 1.0
        return

    # Stop the poller *and wait for it to finish* before writing the
    # terminal state, so it cannot race a stale sub-1.0 progress over the
    # succeeded row.
    await _stop(poller)

    base = f"/api/v1/jobs/{job_id}"
    preds = build_predictions_from_csv(
        job_id=job_id,
        filename=filename,
        csv_text=resp.csv,
        meta=resp.meta,
        video_url=f"{base}/video",
        csv_url=f"{base}/result.csv",
    )
    async with session_scope() as s:
        job = await s.get(_JobModel, job_id)
        if job is None:
            return
        job.status = JobStatus.succeeded.value
        job.progress = 1.0
        job.rows = resp.rows
        job.result_csv = resp.csv
        job.predictions_json = preds.model_dump_json()
    logger.info("job=%s succeeded rows=%s", job_id, resp.rows)


async def _get_job_or_404(s: AsyncSession, job_id: UUID) -> _JobModel:
    job = await s.get(_JobModel, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")
    return job


# ===========================================================================
# Routes
# ===========================================================================


@router.post("", status_code=status.HTTP_201_CREATED, response_model=JobResponse)
async def create_job(video: UploadFile) -> JobResponse:
    if settings.MOCK_MODE:
        return _mock_create(video)

    job_id = uuid4()
    filename, video_path = _save_upload(video, job_id)
    async with session_scope() as s:
        s.add(
            _JobModel(
                id=job_id,
                status=JobStatus.queued.value,
                progress=0.0,
                filename=filename,
                video_path=str(video_path),
            )
        )
    task = asyncio.create_task(
        _process_job(job_id, str(video_path), filename)
    )
    _BG_TASKS.add(task)
    task.add_done_callback(_BG_TASKS.discard)

    return JobResponse(
        id=job_id,
        status=JobStatus.queued,
        progress=0.0,
        filename=filename,
        created_at=_now(),
        updated_at=_now(),
    )


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(job_id: UUID) -> JobResponse:
    if settings.MOCK_MODE:
        return _mock_get(job_id)

    # The cache only ever holds *terminal* (immutable) job responses, so a
    # hit is always safe and skips the DB entirely. A running job is never
    # cached — its progress changes every poll (see app/cache/redis.py).
    cache = get_redis_client()
    key = f"job:{job_id}"
    cached = await cache.get_json(key)
    if cached is not None:
        return JobResponse.model_validate(cached)

    async with session_scope() as s:
        job = await _get_job_or_404(s, job_id)
        resp = _job_response(job)
        terminal = job.status in (
            JobStatus.succeeded.value,
            JobStatus.failed.value,
        )
    if terminal:
        await cache.set_json(
            key, resp.model_dump(mode="json"), settings.CACHE_DEFAULT_TTL
        )
    return resp


@router.get("/{job_id}/video")
async def get_video(job_id: UUID) -> FileResponse:
    if settings.MOCK_MODE:
        video_path: Path = _mock_get_or_404(job_id)["video_path"]
    else:
        async with session_scope() as s:
            video_path = Path((await _get_job_or_404(s, job_id)).video_path)
    if not video_path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Source video not found")
    # FileResponse honours HTTP Range so the SPA can seek to any tag's
    # timestamp without downloading the whole clip.
    return FileResponse(video_path, filename=f"source{video_path.suffix}")


@router.get("/{job_id}/predictions", response_model=JobPredictions)
async def get_predictions(job_id: UUID) -> JobPredictions:
    if settings.MOCK_MODE:
        job = _mock_get_or_404(job_id)
        if job["status"] != JobStatus.succeeded:
            raise HTTPException(status.HTTP_409_CONFLICT, "Predictions not ready")
        base = f"/api/v1/jobs/{job_id}"
        return build_predictions(
            job_id=job_id,
            filename=job["filename"],
            tags=_mock_tags_for(job),
            video_url=f"{base}/video",
            csv_url=f"{base}/result.csv",
        )

    async with session_scope() as s:
        job = await _get_job_or_404(s, job_id)
        if job.status != JobStatus.succeeded.value or not job.predictions_json:
            raise HTTPException(status.HTTP_409_CONFLICT, "Predictions not ready")
        return JobPredictions.model_validate_json(job.predictions_json)


@router.get("/{job_id}/result.csv", response_class=PlainTextResponse)
async def download_result(job_id: UUID) -> PlainTextResponse:
    if settings.MOCK_MODE:
        job = _mock_get_or_404(job_id)
        if job["status"] != JobStatus.succeeded:
            raise HTTPException(status.HTTP_409_CONFLICT, "Result not ready")
        csv_text = build_csv(_mock_tags_for(job))
    else:
        async with session_scope() as s:
            job = await _get_job_or_404(s, job_id)
            if job.status != JobStatus.succeeded.value or job.result_csv is None:
                raise HTTPException(status.HTTP_409_CONFLICT, "Result not ready")
            csv_text = job.result_csv

    return PlainTextResponse(
        csv_text,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="result.csv"'},
    )
