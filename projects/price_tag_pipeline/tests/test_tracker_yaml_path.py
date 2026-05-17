"""Tracker-YAML resolution tests.

Regression guard for the bug where a bare ``tracker_yaml: botsort.yaml`` made
Ultralytics load its OWN stock config and silently ignore the project's tuned
tracker (every tracker-tuning change a no-op). These tests are offline and need
no model/ultralytics — pure path logic against the committed
``configs/trackers/`` files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from price_tag_pipeline.detector import resolve_tracker_yaml  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRACKERS = PROJECT_ROOT / "configs" / "trackers"


def test_explicit_project_path_resolves_to_absolute_file():
    resolved = Path(resolve_tracker_yaml("configs/trackers/botsort.yaml"))
    assert resolved.is_file()
    assert resolved == (TRACKERS / "botsort.yaml").resolve()


def test_bare_botsort_upgrades_to_tuned_project_file():
    # The actual bug fix: a bare builtin name must NOT fall through to
    # Ultralytics' stock config when a same-named tuned file is shipped.
    assert Path(resolve_tracker_yaml("botsort.yaml")) == (
        TRACKERS / "botsort.yaml"
    ).resolve()


def test_bare_bytetrack_upgrades_to_tuned_project_file():
    assert Path(resolve_tracker_yaml("bytetrack.yaml")) == (
        TRACKERS / "bytetrack.yaml"
    ).resolve()


def test_unknown_name_fails_fast_no_silent_stock_fallback():
    with pytest.raises(FileNotFoundError, match="not found"):
        resolve_tracker_yaml("does-not-exist.yaml")


def test_empty_value_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        resolve_tracker_yaml("   ")


def test_absolute_existing_path_passes_through_resolved():
    abs_path = (TRACKERS / "botsort.yaml").resolve()
    assert Path(resolve_tracker_yaml(str(abs_path))) == abs_path
