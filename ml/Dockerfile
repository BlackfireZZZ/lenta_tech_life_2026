# ML inference service — REAL GPU image.
#
# Build context is the repo ROOT, not ./ml (docker-compose.yaml:
#   build: { context: ., dockerfile: ml/Dockerfile }).
# Reason: app/runner.py imports the model from
# ../projects/price_tag_pipeline/src via sys.path, so the package must be
# inside the build context. The in-container layout mirrors the repo
# (/workspace/ml + /workspace/projects/...) so runner.py's
# `Path(__file__).parents[2]` still resolves to the repo root.
#
# Model WEIGHTS are NOT baked into the image. Qwen3-VL-4B (~8 GB) and the
# OpenFoodFacts detector resolve through the Hugging Face hub cache at
# $HF_HOME, which docker-compose mounts as the `hf_cache` volume — downloaded
# once on first run, persisted across rebuilds, never in an image layer.
# The image carries only deps + code.
#
# Mock fallback is preserved: with ML_MOCK=1 (or if a heavy import fails)
# runner.py serves the deterministic mock CSV and the service still boots.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models/hf \
    HF_HUB_ENABLE_HF_TRANSFER=1

# System deps:
#   ffmpeg + libgl1 + libglib2.0-0  — OpenCV / video decode
#   tesseract-ocr(-rus)             — pytesseract OCR fallback
#   libzbar0                        — pyzbar (1D barcode / QR decode)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip python3.11-dev \
        build-essential git curl ca-certificates \
        ffmpeg libgl1 libglib2.0-0 \
        tesseract-ocr tesseract-ocr-rus \
        libzbar0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python3 && \
    python -m pip install --upgrade pip wheel setuptools

WORKDIR /workspace

# 1) Torch with CUDA 12.4 wheels FIRST, so the later `torch>=2.3` pin in
#    ocr.txt resolves to this CUDA build instead of pulling a CPU wheel.
#    Same recipe as the root training Dockerfile.
RUN pip install --index-url https://download.pytorch.org/whl/cu124 \
        torch torchvision

# 2) Pipeline runtime deps (cached on the requirements layer).
#    base.txt = detector/tracking core; ocr.txt = the Qwen3-VL transformers
#    stack (+ classical PaddleOCR fallback). hf_transfer honours
#    HF_HUB_ENABLE_HF_TRANSFER (fast weight download). zxing-cpp + pyzbar
#    are the QR/1D-barcode decoders (optional in code, installed here so the
#    real recognition chain is fully capable).
COPY projects/price_tag_pipeline/requirements ./projects/price_tag_pipeline/requirements
RUN pip install \
        -r projects/price_tag_pipeline/requirements/base.txt \
        -r projects/price_tag_pipeline/requirements/ocr.txt \
        hf_transfer zxing-cpp pyzbar

# 3) The thin service deps.
RUN pip install fastapi "uvicorn[standard]" "pydantic>=2"

# 4) Code only. Weights, datasets, runs/ are excluded by the root
#    .dockerignore — they arrive at runtime via mounted volumes.
COPY projects/price_tag_pipeline ./projects/price_tag_pipeline
COPY ml ./ml

# Import smoke — non-fatal on purpose: the service must boot even if a heavy
# import fails (runner.py then serves the guarded mock, architecture.md §5).
RUN python -c "import torch; print('torch', torch.__version__, 'cuda_build', torch.version.cuda)" ; \
    python -c "import sys; sys.path.insert(0, 'projects/price_tag_pipeline/src'); import price_tag_pipeline.pipeline, price_tag_pipeline.submission; print('pipeline import ok')" ; \
    true

WORKDIR /workspace/ml
EXPOSE 8002
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
