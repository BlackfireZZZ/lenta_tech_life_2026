"""Single-worker serialization of video processing (the GPU is the bottleneck).

**Why this exists.** The product runs on a deliberately *cheap* rented box
with one small GPU. A single ML run (detector + Qwen-VL OCR over a whole
clip) already saturates that GPU's VRAM. Two clips in flight at once would
thrash VRAM — both runs get dramatically slower, or the second OOMs and
fails. So uploaded videos are processed **strictly one at a time, FIFO**: a
second upload sits in ``queued`` ("В очереди") until the one ahead of it
finishes, and the user is shown their place in line.

This replaces the old "one ``asyncio`` task per upload" model in
``routes/jobs.py`` (unbounded parallelism — the exact thing that kills a
small GPU). The HTTP contract is unchanged: ``POST /jobs`` still returns
immediately with a ``queued`` job (architecture.md §5.3); only *when* it
starts running changed.

**Durable across a restart.** The asyncio queue is in-process (one ML
container, one box — architecture.md §5.5), so on boot it is rebuilt from
Postgres: any ``running`` row is a job whose worker died mid-flight → reset
to ``queued``; then every ``queued`` row is re-enqueued in ``created_at``
order. Postgres stays the source of truth; the queue is just the scheduler.
If the product were ever scaled to several GPU boxes this single in-process
worker is the one piece that would become a shared broker — nothing else
changes (architecture.md §5.3/§8).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from typing import Protocol
from uuid import UUID

from sqlalchemy import select, update

from app.api.v1.schemas.job import JobStatus
from app.core.logger import logger
from app.db.models.job import Job as _JobModel
from app.db.session import session_scope
from app.ml.client import ml_client
from app.ml.schemas import ProcessRequest
from app.predictions import build_predictions_from_csv

_NONTERMINAL = (JobStatus.queued.value, JobStatus.running.value)


# ---------------------------------------------------------------------------
# Queue-position (pure — unit-tested without a DB)
# ---------------------------------------------------------------------------


class _JobLike(Protocol):
    id: UUID
    status: str
    created_at: object  # any orderable (datetime); only `<` is used


def position_in_queue(jobs: Iterable[_JobLike], job_id: UUID) -> int | None:
    """How many videos are processed *before* ``job_id`` (FIFO, 1 worker).

    ``jobs`` is any iterable of rows exposing ``id``/``status``/``created_at``
    (a SQLAlchemy ``Row`` or the model). Returns:

    * ``None`` — the job is terminal (succeeded/failed) or not present;
    * ``0``    — it is the one currently *running*, **or** it is next up
      with nothing ahead of it;
    * ``N>0``  — ``N`` jobs (the running one + earlier-queued ones) will be
      processed before it.

    The caller decides the wording from the job's *status* (running vs
    queued); this only supplies the number.
    """
    target = next((j for j in jobs if j.id == job_id), None)
    if target is None or target.status not in _NONTERMINAL:
        return None
    if target.status == JobStatus.running.value:
        return 0
    ahead = 0
    for j in jobs:
        if j.id == target.id:
            continue
        if j.status == JobStatus.running.value:
            ahead += 1
        elif (
            j.status == JobStatus.queued.value
            and j.created_at < target.created_at
        ):
            ahead += 1
    return ahead


# ---------------------------------------------------------------------------
# The out-of-band worker body (moved verbatim from routes/jobs.py, now keyed
# only by job_id — every input is already persisted on the row).
# ---------------------------------------------------------------------------


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
        phase = snap.get("phase") or None
        try:
            async with session_scope() as s:
                job = await s.get(_JobModel, job_id)
                if job is None or job.status != JobStatus.running.value:
                    return
                if value > job.progress:
                    job.progress = value
                # Phase can change even while the fraction barely moves (the
                # long finalize/OCR burst) — persist it independently so the
                # bar's label stays truthful.
                if phase and phase != job.phase:
                    job.phase = phase
        except Exception as exc:  # pragma: no cover - never fatal
            logger.debug("progress write failed (job=%s): %s", job_id, exc)


async def _process_job(job_id: UUID) -> None:
    """Run the ML pipeline for one job and persist its result.

    Keyed by id only: ``video_path``/``filename``/``rotation`` are already on
    the row (the request that created it is long gone). Owns its DB sessions.
    Any failure → the job is marked ``failed`` with the message; it never
    crashes the worker (the loop catches and moves to the next job too).
    """
    async with session_scope() as s:
        job = await s.get(_JobModel, job_id)
        if job is None:
            return
        job.status = JobStatus.running.value
        job.progress = 0.0
        job.phase = None
        video_path = job.video_path
        filename = job.filename
        rotation = job.rotation

    poller = asyncio.create_task(_poll_progress(job_id))
    try:
        resp = await ml_client.process(
            ProcessRequest(
                video_path=video_path,
                job_id=str(job_id),
                filename=filename,
                rotation=rotation,
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


# ---------------------------------------------------------------------------
# The serializer
# ---------------------------------------------------------------------------

Processor = Callable[[UUID], Awaitable[None]]


class JobQueue:
    """A single long-lived consumer of an unbounded ``asyncio.Queue``.

    ``enqueue`` only ever appends; the one worker coroutine awaits each
    ``processor(job_id)`` to completion before taking the next id — that is
    the entire "one video at a time" guarantee. ``processor`` is injectable
    so the serialization can be unit-tested without a DB/GPU.
    """

    def __init__(self, processor: Processor = _process_job) -> None:
        self._processor = processor
        self._queue: asyncio.Queue[UUID] = asyncio.Queue()
        self._worker: asyncio.Task | None = None

    def enqueue(self, job_id: UUID) -> None:
        """Append a job. The row must already be committed as ``queued`` so
        the worker (a separate session) can see it."""
        self._queue.put_nowait(job_id)

    @property
    def pending(self) -> int:
        """Best-effort count of ids waiting in the in-memory queue (the DB
        is authoritative for what the user is shown — see
        :func:`position_in_queue`)."""
        return self._queue.qsize()

    async def _run(self) -> None:
        while True:
            job_id = await self._queue.get()
            try:
                await self._processor(job_id)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - one bad job must not stop the line
                logger.exception("queue worker: job=%s crashed", job_id)
            finally:
                self._queue.task_done()

    async def _recover_orphans(self) -> None:
        """Rebuild the queue from Postgres on boot (architecture.md §5.5).

        A ``running`` row at startup is a job whose worker died with the
        previous process → reset it to ``queued``. Then enqueue every
        ``queued`` row oldest-first so FIFO order survives a restart.
        """
        async with session_scope() as s:
            await s.execute(
                update(_JobModel)
                .where(_JobModel.status == JobStatus.running.value)
                .values(
                    status=JobStatus.queued.value, progress=0.0, phase=None
                )
            )
            ids = (
                await s.execute(
                    select(_JobModel.id)
                    .where(_JobModel.status == JobStatus.queued.value)
                    .order_by(_JobModel.created_at.asc())
                )
            ).scalars().all()
        for jid in ids:
            self._queue.put_nowait(jid)
        if ids:
            logger.info("requeued %d unfinished job(s) after restart", len(ids))

    async def start(self) -> None:
        """Spawn the single worker, then re-enqueue any unfinished jobs."""
        if self._worker is not None:
            return
        self._worker = asyncio.create_task(self._run())
        await self._recover_orphans()
        logger.info("job queue started (serialized — 1 video at a time)")

    async def stop(self) -> None:
        if self._worker is not None:
            await _stop(self._worker)
            self._worker = None


# One instance per app (mirrors ml_client). Wired in the lifespan.
job_queue = JobQueue()
