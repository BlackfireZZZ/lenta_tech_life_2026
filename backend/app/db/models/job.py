"""`jobs` table — one uploaded video → one CSV result.

PLACEHOLDER (docs/architecture.md §3.4). One entity = one file. UUID PK,
``created_at``/``updated_at`` via ``server_default=func.now()``. Implement
when persistence replaces the in-memory mock store in routes/jobs.py.
"""

# from sqlalchemy import Column, DateTime, Integer, String, Text, func
# from sqlalchemy.dialects.postgresql import UUID
# from app.db.base import Base
#
# class Job(Base):
#     __tablename__ = "jobs"
#     id = Column(UUID(as_uuid=True), primary_key=True, index=True)
#     status = Column(String, nullable=False, index=True)   # queued/running/...
#     progress = Column(Integer, nullable=False, default=0)  # 0..100
#     filename = Column(String, nullable=False)
#     rows = Column(Integer, nullable=True)
#     error = Column(Text, nullable=True)
#     result_csv = Column(Text, nullable=True)               # graded 29-col CSV
#     created_at = Column(DateTime(timezone=True), server_default=func.now())
#     updated_at = Column(DateTime(timezone=True),
#                         server_default=func.now(), onupdate=func.now())
