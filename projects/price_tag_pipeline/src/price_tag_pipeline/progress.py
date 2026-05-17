"""Progress reporting for long-running video inference.

A robot video is minutes long; the task spec mandates a progress bar in the
UI (``docs/hackathon/task.md`` §13) and a poll-able job status in the product
(``docs/architecture.md`` §3.4 / §5.3). Both need the *same* signal from one
place: the inference loop. This module is that single source of truth.

Design goals:
- **Zero hard dependencies.** ``tqdm`` is optional; if it is missing the
  CLI bar silently degrades to log lines. The pipeline never imports tqdm.
- **One event type.** ``ProgressEvent`` carries everything any consumer
  needs (a tqdm bar, a Gradio ``gr.Progress``, an ML-service registry the
  gateway can poll). Consumers map the *same* event to their own UI.
- **Cheap and safe.** Reporting must never slow down or crash the run; a
  failing reporter is logged once and then ignored.

The pipeline only ever calls :meth:`ProgressReporter.publish`. Everything
else here is for consumers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional, Union

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phases — coarse stages of a single video run, in order.
# ---------------------------------------------------------------------------

class Phase:
    """Ordered, stable phase identifiers (stable: consumers may switch on them)."""

    DETECT = "detect"      # per-frame detect + track + buffer crops (the bulk)
    FINALIZE = "finalize"  # OCR best crops of tracks still live at video end
    DEDUP = "dedup"        # cross-track deduplication on the full tag list
    DONE = "done"          # finished; fraction == 1.0


@dataclass(frozen=True)
class ProgressEvent:
    """One immutable progress snapshot.

    ``fraction`` is the single number a bar should render; it is monotonic
    non-decreasing across a run and always in ``[0.0, 1.0]``. The remaining
    fields are context a richer UI can show (frame counters, tag count).
    """

    phase: str
    fraction: float
    frames_done: int = 0
    frames_total: int = 0           # 0 == unknown (container did not report it)
    tags_finalized: int = 0
    message: str = ""

    def as_dict(self) -> dict:
        """JSON-friendly form (used by the ML-service progress side-channel)."""
        return {
            "phase": self.phase,
            "fraction": round(float(self.fraction), 4),
            "frames_done": int(self.frames_done),
            "frames_total": int(self.frames_total),
            "tags_finalized": int(self.tags_finalized),
            "message": self.message,
        }


# ---------------------------------------------------------------------------
# Reporters
# ---------------------------------------------------------------------------

class ProgressReporter:
    """Base reporter. Subclasses override :meth:`_emit`.

    :meth:`publish` is the only method the pipeline calls. It clamps the
    fraction to ``[0, 1]``, enforces monotonicity (a noisy estimate must
    never make the bar go backwards), and isolates consumer failures so a
    broken UI can never abort a multi-minute run.
    """

    def __init__(self) -> None:
        self._last_fraction: float = 0.0
        self._broken: bool = False

    def publish(self, event: ProgressEvent) -> None:
        frac = min(1.0, max(0.0, float(event.fraction)))
        if frac < self._last_fraction:
            frac = self._last_fraction
        self._last_fraction = frac
        if frac != event.fraction:
            event = ProgressEvent(
                phase=event.phase,
                fraction=frac,
                frames_done=event.frames_done,
                frames_total=event.frames_total,
                tags_finalized=event.tags_finalized,
                message=event.message,
            )
        if self._broken:
            return
        try:
            self._emit(event)
        except Exception as exc:  # a broken bar must not kill inference
            LOGGER.warning("Progress reporter failed (disabled for this run): %s", exc)
            self._broken = True

    def close(self) -> None:
        """Release any consumer resources (e.g. close a tqdm bar)."""

    # -- override this --
    def _emit(self, event: ProgressEvent) -> None:  # pragma: no cover - base
        raise NotImplementedError


class NullProgress(ProgressReporter):
    """Default no-op reporter — what existing callers get for free."""

    def _emit(self, event: ProgressEvent) -> None:
        return


class CallbackProgress(ProgressReporter):
    """Forward every (clamped) event to a plain callable.

    Used by the Gradio UI (``lambda e: gr.Progress()(e.fraction, e.message)``)
    and by the ML service (records the event into a job→progress registry the
    gateway polls).
    """

    def __init__(self, callback: Callable[[ProgressEvent], None]) -> None:
        super().__init__()
        self._callback = callback

    def _emit(self, event: ProgressEvent) -> None:
        self._callback(event)


class TqdmProgress(ProgressReporter):
    """A terminal bar for the CLI. ``tqdm`` is optional.

    If tqdm is not installed the bar degrades to one INFO log line per phase
    change plus a final summary — the pipeline still runs identically.
    """

    def __init__(self) -> None:
        super().__init__()
        self._bar = None
        self._phase: str = ""
        try:
            from tqdm import tqdm  # type: ignore

            self._tqdm = tqdm
        except Exception:
            self._tqdm = None
            LOGGER.info("tqdm not installed — progress shown as log lines only.")

    def _emit(self, event: ProgressEvent) -> None:
        if self._tqdm is None:
            if event.phase != self._phase:
                self._phase = event.phase
                LOGGER.info(
                    "[%s] %.0f%% (%d/%d frames, %d tags) %s",
                    event.phase,
                    event.fraction * 100.0,
                    event.frames_done,
                    event.frames_total,
                    event.tags_finalized,
                    event.message,
                )
            return

        if self._bar is None:
            self._bar = self._tqdm(
                total=1000, unit="‰", bar_format="{l_bar}{bar}| {postfix}"
            )
        self._bar.n = int(event.fraction * 1000)
        self._bar.set_description_str(event.phase)
        self._bar.set_postfix_str(
            f"{event.frames_done}/{event.frames_total}f "
            f"{event.tags_finalized}tags {event.message}".strip()
        )
        self._bar.refresh()

    def close(self) -> None:
        if self._bar is not None:
            self._bar.n = 1000
            self._bar.refresh()
            self._bar.close()
            self._bar = None


# ---------------------------------------------------------------------------
# Coercion — keep ``PriceTagPipeline.run(progress=...)`` ergonomic.
# ---------------------------------------------------------------------------

ProgressLike = Union[ProgressReporter, Callable[[ProgressEvent], None], None]


def as_reporter(progress: ProgressLike) -> ProgressReporter:
    """Accept ``None`` | a reporter | a bare callback → always a reporter.

    Lets callers pass ``progress=lambda e: ...`` without importing a class,
    while the pipeline only ever deals with the uniform reporter interface.
    """
    if progress is None:
        return NullProgress()
    if isinstance(progress, ProgressReporter):
        return progress
    if callable(progress):
        return CallbackProgress(progress)
    raise TypeError(
        f"progress must be None, a ProgressReporter, or callable; got {type(progress)!r}"
    )
