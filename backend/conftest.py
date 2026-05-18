"""Make ``import app...`` (backend) and ``price_tag_pipeline`` (the model
package) importable in tests without installing either — same sys.path
convention the ML service and pipeline scripts use.
"""

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent
_PIPELINE_SRC = _BACKEND.parent / "projects" / "price_tag_pipeline" / "src"

for p in (_BACKEND, _PIPELINE_SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
