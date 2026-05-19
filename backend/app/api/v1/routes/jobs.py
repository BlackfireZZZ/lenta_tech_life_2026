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
  request returns immediately with a ``queued`` job and a single FIFO
  worker (``app/jobs_queue.py``) runs them **one at a time** (the rented
  GPU only fits one clip), polling ML ``/progress`` to drive
  ``JobResponse.progress``, then persisting the verbatim graded CSV and the
  reconstructed review payload. While a job waits its turn the response
  carries ``queue_position`` (how many videos are ahead of it).

The graded CSV is served byte-for-byte as the ML service produced it — the
gateway never reshapes graded columns.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from tempfile import gettempdir
from uuid import UUID, uuid4

from fastapi import APIRouter, Form, HTTPException, UploadFile, status
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    PlainTextResponse,
    Response,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.schemas.job import JobPredictions, JobResponse, JobStatus
from app.cache.redis import get_redis_client
from app.core.config import settings
from app.core.logger import logger
from app.db.models.job import Job as _JobModel
from app.db.session import session_scope
from app.jobs_mock import (
    build_csv,
    build_detections,
    build_predictions,
    generate_tags,
)
from app.jobs_queue import job_queue, position_in_queue

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


def _save_upload(video: UploadFile, job_id: UUID) -> tuple[str, Path, str]:
    """Persist the clip under an ASCII-safe name; return
    (orig_name, path, sha256). The original name may be non-ASCII; keep it
    only as metadata and store the bytes as ``source.<ext>`` so non-ASCII
    paths never break I/O. The hash is computed in the same streaming pass
    (no second read) and is the content-cache key.
    """
    filename = video.filename or "upload.mp4"
    ext = Path(filename).suffix.lower() or ".mp4"
    video_path = _upload_dir(job_id) / f"source{ext}"
    digest = hashlib.sha256()
    with video_path.open("wb") as out:
        while True:
            chunk = video.file.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            digest.update(chunk)
    return filename, video_path, digest.hexdigest()


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
    filename, video_path, _ = _save_upload(video, job_id)
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

def _job_response(job, queue_position: int | None = None) -> JobResponse:
    base = f"/api/v1/jobs/{job.id}"
    done = job.status == JobStatus.succeeded.value
    return JobResponse(
        id=job.id,
        status=JobStatus(job.status),
        progress=job.progress,
        filename=job.filename,
        rows=job.rows,
        error=job.error,
        phase=job.phase,
        # Only meaningful while non-terminal: how many videos the single
        # worker must finish before this one (0 = running / next up).
        queue_position=queue_position,
        result_csv_url=f"{base}/result.csv" if done else None,
        predictions_url=f"{base}/predictions" if done else None,
        video_url=f"{base}/video" if done else None,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


async def _nonterminal_rows(s: AsyncSession) -> list:
    """The (small) set of queued/running jobs — the input to
    :func:`position_in_queue`. There is at most one running (single worker)
    and rarely more than a handful waiting, so this stays cheap."""
    return (
        await s.execute(
            select(
                _JobModel.id, _JobModel.status, _JobModel.created_at
            ).where(
                _JobModel.status.in_(
                    (JobStatus.queued.value, JobStatus.running.value)
                )
            )
        )
    ).all()


ROTATIONS = {"none", "ccw", "cw"}


def _norm_rotation(value: str | None) -> str:
    """Detector-only pre-rotation. Unknown/empty → 'none': an uploaded clip
    is trusted to be in its real-life orientation and passes through the
    detector untouched (verified: a normal upright phone clip gets clean
    boxes at 'none', garbage at 'ccw'). The UI rotate button is the per-clip
    override for sideways footage (e.g. the robot cam mounted 90° CW)."""
    v = (value or "none").strip().lower()
    return v if v in ROTATIONS else "none"


MODES = {"full", "fast"}


def _norm_mode(value: str | None) -> str:
    """Recognition depth from the UI checkbox. Unknown/empty → 'full' (the
    canonical balanced.yaml behaviour — every tag's top-K sharpest crops go
    through Qwen3-VL). 'fast' makes the ML service cap that at the single
    sharpest crop per tag: ~5× fewer (slow) VLM calls, a bit less voting
    redundancy. Detection/tracking are never cut — they are cheap."""
    v = (value or "full").strip().lower()
    return v if v in MODES else "full"


async def _get_job_or_404(s: AsyncSession, job_id: UUID) -> _JobModel:
    job = await s.get(_JobModel, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")
    return job


async def _find_cached(
    s: AsyncSession, content_hash: str, rotation: str, mode: str
) -> _JobModel | None:
    """Newest succeeded job with the same bytes + rotation + mode, complete
    enough to replay. Rotation changes detection and mode changes how many
    crops the VLM votes on — both change the output, so both are part of the
    key (a fast run must never be served for a full request, or vice versa)."""
    stmt = (
        select(_JobModel)
        .where(
            _JobModel.content_hash == content_hash,
            _JobModel.rotation == rotation,
            _JobModel.mode == mode,
            _JobModel.status == JobStatus.succeeded.value,
            _JobModel.result_csv.is_not(None),
            _JobModel.predictions_json.is_not(None),
        )
        .order_by(_JobModel.created_at.desc())
        .limit(1)
    )
    return (await s.execute(stmt)).scalars().first()


# ===========================================================================
# Routes
# ===========================================================================


@router.get("", response_model=list[JobResponse])
async def list_jobs() -> list[JobResponse]:
    """Every job, newest first — the user's way back to earlier work.
    Capped so the list stays cheap; the UI only needs recent history."""
    if settings.MOCK_MODE:
        jobs = sorted(
            _MOCK_JOBS.values(), key=lambda j: j["created_at"], reverse=True
        )
        return [JobResponse.model_validate(j) for j in jobs]
    async with session_scope() as s:
        rows = (
            await s.execute(
                select(_JobModel)
                .order_by(_JobModel.created_at.desc())
                .limit(100)
            )
        ).scalars().all()
        # One small query for the queued/running set, reused for every row's
        # place-in-line (terminal rows → None).
        pending = await _nonterminal_rows(s)
        return [
            _job_response(j, position_in_queue(pending, j.id)) for j in rows
        ]


@router.post("", status_code=status.HTTP_201_CREATED, response_model=JobResponse)
async def create_job(
    video: UploadFile,
    rotation: str = Form("none"),
    mode: str = Form("full"),
) -> JobResponse:
    if settings.MOCK_MODE:
        return _mock_create(video)

    rotation = _norm_rotation(rotation)
    mode = _norm_mode(mode)
    job_id = uuid4()
    filename, video_path, content_hash = _save_upload(video, job_id)
    base = f"/api/v1/jobs/{job_id}"

    async with session_scope() as s:
        cached = await _find_cached(s, content_hash, rotation, mode)
        if cached is not None:
            # Replay a prior identical run instantly (no pipeline). The new
            # job has its own stored clip, so /video works; we only repoint
            # the predictions payload's ids/urls at this job.
            preds = cached.predictions_json
            try:
                jp = JobPredictions.model_validate_json(preds)
                jp.job_id = job_id
                jp.video_url = f"{base}/video"
                jp.csv_url = f"{base}/result.csv"
                preds = jp.model_dump_json()
            except Exception:  # malformed cache → UI builds urls itself
                pass
            s.add(
                _JobModel(
                    id=job_id,
                    status=JobStatus.succeeded.value,
                    progress=1.0,
                    filename=filename,
                    video_path=str(video_path),
                    rotation=rotation,
                    mode=mode,
                    content_hash=content_hash,
                    rows=cached.rows,
                    result_csv=cached.result_csv,
                    predictions_json=preds,
                    # Carry the detector trace too, so a cache-replayed job
                    # keeps the detector view (it is keyed on the same
                    # bytes+rotation+mode, so the trace is identical).
                    detections_json=cached.detections_json,
                )
            )
            logger.info(
                "job=%s cache HIT (hash=%s… rot=%s mode=%s) reused job=%s",
                job_id, content_hash[:12], rotation, mode, cached.id,
            )
            return JobResponse(
                id=job_id,
                status=JobStatus.succeeded,
                progress=1.0,
                filename=filename,
                rows=cached.rows,
                result_csv_url=f"{base}/result.csv",
                predictions_url=f"{base}/predictions",
                video_url=f"{base}/video",
                created_at=_now(),
                updated_at=_now(),
            )

        s.add(
            _JobModel(
                id=job_id,
                status=JobStatus.queued.value,
                progress=0.0,
                filename=filename,
                video_path=str(video_path),
                rotation=rotation,
                mode=mode,
                content_hash=content_hash,
            )
        )

    # The `queued` row is committed (the `async with` block above exited);
    # hand the id to the single FIFO worker. It will not run until every
    # earlier video finishes — the rented GPU only fits one at a time
    # (app/jobs_queue.py). The user sees its place in line on the next poll.
    job_queue.enqueue(job_id)

    async with session_scope() as s:
        position = position_in_queue(await _nonterminal_rows(s), job_id)

    return JobResponse(
        id=job_id,
        status=JobStatus.queued,
        progress=0.0,
        filename=filename,
        queue_position=position,
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
        terminal = job.status in (
            JobStatus.succeeded.value,
            JobStatus.failed.value,
        )
        position = (
            None
            if terminal
            else position_in_queue(await _nonterminal_rows(s), job_id)
        )
        resp = _job_response(job, position)
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


@router.get("/{job_id}/detections")
async def get_detections(job_id: UUID) -> Response:
    """Per-frame detector trace — the NON-graded detector-QA overlay source.

    Lets the review screen replay the clip with the *raw per-frame detector
    output* overlaid (every box above the detector's confidence threshold),
    so the detector can be judged on its own — separate from the final
    per-tag result the graded CSV carries.

    409 when not ready, or when this job simply has no trace (an older job,
    the ML service sent none, or its guarded collection failed). The graded
    CSV and the best-frame review never depend on this — the UI just offers
    only the best-frame view in that case.
    """
    if settings.MOCK_MODE:
        job = _mock_get_or_404(job_id)
        if job["status"] != JobStatus.succeeded:
            raise HTTPException(
                status.HTTP_409_CONFLICT, "Detections not ready"
            )
        return JSONResponse(build_detections(_mock_tags_for(job)))

    async with session_scope() as s:
        job = await _get_job_or_404(s, job_id)
        if job.status != JobStatus.succeeded.value:
            raise HTTPException(
                status.HTTP_409_CONFLICT, "Detections not ready"
            )
        payload = job.detections_json
    if not payload:
        raise HTTPException(
            status.HTTP_409_CONFLICT, "No detector trace for this job"
        )
    # Stored already as a JSON string — serve verbatim, no re-encode.
    return Response(content=payload, media_type="application/json")


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
