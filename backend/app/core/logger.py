"""Project logger. Kept trivial on purpose — no observability stack on the
hackathon (see docs/architecture.md §8 "what we deliberately drop").
"""

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger("gateway")
