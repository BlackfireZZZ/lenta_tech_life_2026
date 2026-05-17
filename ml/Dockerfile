# ML service image.
#
# MOCK build (default): tiny CPU image, no model deps — boots the contract.
# REAL build: switch the base to the CUDA image and install the pipeline:
#   FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
#   COPY ../projects/price_tag_pipeline /opt/price_tag_pipeline
#   RUN pip install -e /opt/price_tag_pipeline
# (see the root Dockerfile + docs/pipeline-reference.md for the GPU stack).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /srv

RUN pip install --no-cache-dir fastapi "uvicorn[standard]" pydantic

COPY app ./app

EXPOSE 8002
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
