"""Make ``app`` (the ml service package) and ``price_tag_pipeline``
importable in ml tests without installing either — the same sys.path
convention runner.py uses at runtime.

Run ml tests on their own (`pytest ml/tests`): the backend also uses a
top-level ``app`` package, so the two suites must not share a process.
"""

import sys
from pathlib import Path

_ML = Path(__file__).resolve().parent
_PIPELINE_SRC = _ML.parent / "projects" / "price_tag_pipeline" / "src"

for p in (_ML, _PIPELINE_SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
