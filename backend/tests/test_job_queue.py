"""The serializer that protects the cheap rented GPU (app/jobs_queue.py).

Two guarantees, both provable without a DB / GPU / network:

1. ``position_in_queue`` — the number the UI shows ("перед вами N").
2. The worker processes **strictly one job at a time, FIFO**, and one
   crashing job does not stop the line. This is the whole point: two ML
   runs at once would thrash the single small GPU.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.jobs_queue import JobQueue, position_in_queue

_T0 = datetime(2026, 5, 19, tzinfo=timezone.utc)


def _row(status: str, age_s: int):
    """A stand-in for a SQLAlchemy Row / Job (only id/status/created_at)."""
    return SimpleNamespace(
        id=uuid4(), status=status, created_at=_T0 + timedelta(seconds=age_s)
    )


# --- position_in_queue (pure) ---------------------------------------------


def test_running_job_is_position_zero():
    r = _row("running", 0)
    assert position_in_queue([r], r.id) == 0


def test_terminal_or_unknown_job_has_no_position():
    done = _row("succeeded", 0)
    assert position_in_queue([done], done.id) is None
    assert position_in_queue([], uuid4()) is None


def test_queued_counts_the_running_one_plus_earlier_queued():
    running = _row("running", 0)
    q1 = _row("queued", 1)  # oldest waiting
    q2 = _row("queued", 2)
    q3 = _row("queued", 3)
    rows = [q3, running, q1, q2]  # order in the list must not matter

    # q1: only the running job is ahead.
    assert position_in_queue(rows, q1.id) == 1
    # q2: running + q1.
    assert position_in_queue(rows, q2.id) == 2
    # q3: running + q1 + q2.
    assert position_in_queue(rows, q3.id) == 3


def test_next_up_with_nothing_running_is_zero():
    # No running job yet (worker between jobs) → the oldest queued is next.
    q1 = _row("queued", 1)
    q2 = _row("queued", 2)
    assert position_in_queue([q1, q2], q1.id) == 0
    assert position_in_queue([q1, q2], q2.id) == 1


# --- the serializer (real asyncio, injected processor) --------------------


@pytest.mark.asyncio
async def test_worker_processes_strictly_one_at_a_time_in_fifo_order():
    live = 0
    peak = 0
    done: list = []

    async def proc(job_id):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.02)  # overlap window if it were concurrent
        done.append(job_id)
        live -= 1

    jq = JobQueue(processor=proc)
    # Start only the worker; .start() also does DB orphan-recovery, which is
    # out of scope for this unit (covered by the route/integration path).
    jq._worker = asyncio.create_task(jq._run())
    ids = [uuid4() for _ in range(6)]
    for jid in ids:
        jq.enqueue(jid)

    await asyncio.wait_for(jq._queue.join(), timeout=5)
    await jq.stop()

    assert peak == 1, "two videos ran at once — the GPU guard is broken"
    assert done == ids, "jobs must run in submission (FIFO) order"


@pytest.mark.asyncio
async def test_a_crashing_job_does_not_stop_the_line():
    seen: list = []

    async def proc(job_id):
        seen.append(job_id)
        if len(seen) == 1:
            raise RuntimeError("boom")  # first job blows up

    jq = JobQueue(processor=proc)
    jq._worker = asyncio.create_task(jq._run())
    a, b = uuid4(), uuid4()
    jq.enqueue(a)
    jq.enqueue(b)

    await asyncio.wait_for(jq._queue.join(), timeout=5)
    await jq.stop()

    assert seen == [a, b], "the worker must survive a failed job"
