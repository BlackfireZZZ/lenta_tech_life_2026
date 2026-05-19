"""The fast/full recognition-depth flag (no DB / GPU / network).

`mode` rides the same plumbing as `rotation`: a UI choice → `create_job`
Form field → persisted on the Job row → `ProcessRequest` → the ML runner
maps `fast` to a `top_k_crops_per_track` override. Here we pin the two
pure pieces on the gateway side: the normalizer and the contract default
(the runner's config override needs the pipeline package and is exercised
by the real ML path, not this unit).
"""

from app.api.v1.routes.jobs import MODES, _norm_mode
from app.ml.schemas import ProcessRequest


def test_norm_mode_accepts_the_two_valid_values():
    assert _norm_mode("full") == "full"
    assert _norm_mode("fast") == "fast"
    assert MODES == {"full", "fast"}


def test_norm_mode_is_case_and_whitespace_insensitive():
    assert _norm_mode("  Fast ") == "fast"
    assert _norm_mode("FULL") == "full"


def test_norm_mode_defaults_to_full_when_absent_or_unknown():
    # Unknown / empty / None must never silently downgrade quality — the
    # safe default is the canonical full run.
    assert _norm_mode(None) == "full"
    assert _norm_mode("") == "full"
    assert _norm_mode("turbo") == "full"
    assert _norm_mode("none") == "full"  # a rotation value is not a mode


def test_process_request_defaults_keep_old_clients_on_full():
    # An older gateway that doesn't send `mode` must behave exactly as
    # before: full recognition, no rotation. Backward-compatible contract.
    req = ProcessRequest(video_path="/v.mp4", job_id="j1")
    assert req.mode == "full"
    assert req.rotation == "none"
    assert ProcessRequest(video_path="/v", job_id="j", mode="fast").mode == "fast"
