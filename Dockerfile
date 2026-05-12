# CUDA-ready image for training + inference on rented A100/H100 boxes.
#
# Build:
#     docker build -t lenta-2026:latest .
#
# Run interactively:
#     docker run --rm -it --gpus all \
#         -v $(pwd):/workspace \
#         -v ~/.cache/huggingface:/root/.cache/huggingface \
#         lenta-2026:latest bash
#
# Run a training job:
#     docker run --rm --gpus all -v $(pwd):/workspace lenta-2026:latest \
#         python projects/price_tag_pipeline/scripts/train_detector_yolo.py \
#             --dataset data/processed/dataset.yaml --model yolo26l.pt
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

# System deps. tesseract-ocr-rus + libgl1 are required by pytesseract and cv2.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip python3.11-dev \
        build-essential git curl ca-certificates \
        ffmpeg libgl1 libglib2.0-0 \
        tesseract-ocr tesseract-ocr-rus \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default `python`.
RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python3 && \
    python -m pip install --upgrade pip wheel setuptools

WORKDIR /workspace

# Install Python deps first for layer cache.
COPY projects/price_tag_pipeline/requirements ./projects/price_tag_pipeline/requirements
COPY projects/price_tag_pipeline/requirements.txt ./projects/price_tag_pipeline/requirements.txt

# Pre-install torch with CUDA 12.4 wheels (faster cold start than the default).
RUN pip install --index-url https://download.pytorch.org/whl/cu124 \
        torch torchvision

# Then the project's regular requirements (transformers, ultralytics, etc).
RUN pip install -r projects/price_tag_pipeline/requirements.txt

# Bring in the rest of the repo.
COPY . .

# Smoke-test that imports work.
RUN python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" \
    && python -c "from price_tag_pipeline.config import load_config; print('config ok')" \
        ; true

CMD ["bash"]
