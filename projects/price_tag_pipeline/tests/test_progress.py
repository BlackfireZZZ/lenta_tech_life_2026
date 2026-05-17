"""Progress reporter tests — no GPU / video / model needed."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from price_tag_pipeline.progress import (  # noqa: E402
    CallbackProgress,
    NullProgress,
    Phase,
    ProgressEvent,
    ProgressReporter,
    TqdmProgress,
    as_reporter,
)


def test_event_as_dict_rounds_fraction():
    d = ProgressEvent(phase=Phase.DETECT, fraction=0.123456, frames_done=3,
                       frames_total=10, tags_finalized=1, message="x").as_dict()
    assert d == {
        "phase": "detect", "fraction": 0.1235, "frames_done": 3,
        "frames_total": 10, "tags_finalized": 1, "message": "x",
    }


def test_as_reporter_coercion():
    assert isinstance(as_reporter(None), NullProgress)
    seen: list[ProgressEvent] = []
    cb = as_reporter(seen.append)
    assert isinstance(cb, CallbackProgress)
    existing = NullProgress()
    assert as_reporter(existing) is existing
    with pytest.raises(TypeError):
        as_reporter(123)  # type: ignore[arg-type]


def test_null_progress_is_silent():
    r = NullProgress()
    r.publish(ProgressEvent(phase=Phase.DETECT, fraction=0.5))
    r.close()  # must not raise


def test_callback_clamps_and_is_monotonic():
    seen: list[float] = []
    r = CallbackProgress(lambda e: seen.append(e.fraction))
    r.publish(ProgressEvent(phase=Phase.DETECT, fraction=0.4))
    r.publish(ProgressEvent(phase=Phase.DETECT, fraction=0.2))   # backwards
    r.publish(ProgressEvent(phase=Phase.DETECT, fraction=1.7))   # > 1
    r.publish(ProgressEvent(phase=Phase.DETECT, fraction=-0.5))  # < 0
    assert seen == [0.4, 0.4, 1.0, 1.0]  # never decreases, always in [0,1]


def test_broken_reporter_disables_itself_and_never_raises():
    calls = {"n": 0}

    def boom(_e: ProgressEvent) -> None:
        calls["n"] += 1
        raise RuntimeError("UI exploded")

    r = CallbackProgress(boom)
    r.publish(ProgressEvent(phase=Phase.DETECT, fraction=0.1))  # swallowed
    r.publish(ProgressEvent(phase=Phase.DETECT, fraction=0.2))  # skipped
    assert calls["n"] == 1  # disabled after first failure


def test_tqdm_progress_safe_with_or_without_tqdm():
    r = TqdmProgress()
    assert isinstance(r, ProgressReporter)
    r.publish(ProgressEvent(phase=Phase.DETECT, fraction=0.5,
                            frames_done=5, frames_total=10))
    r.publish(ProgressEvent(phase=Phase.DONE, fraction=1.0))
    r.close()  # must not raise on either branch
