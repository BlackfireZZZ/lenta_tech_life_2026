"""Import every model here so ``Base.metadata`` is fully populated before
``create_all`` runs (docs/architecture.md §3.3). One import per entity.
"""

from app.db.models.job import Job  # noqa: F401

__all__ = ["Job"]
