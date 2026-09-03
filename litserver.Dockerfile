FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Stub src/ so the pip install layer caches on pyproject.toml, not on code
# changes. Real src/ lands via COPY . . and shadows this at import time.
COPY pyproject.toml .
RUN mkdir -p src && touch src/__init__.py
RUN pip install --no-cache-dir --break-system-packages ".[serve]"

ARG SHERPA_ONNX_CUDA_VERSION=""
RUN if [ -n "$SHERPA_ONNX_CUDA_VERSION" ]; then \
        pip install --no-cache-dir --break-system-packages --force-reinstall \
            "sherpa-onnx==${SHERPA_ONNX_CUDA_VERSION}" \
            -f https://k2-fsa.github.io/sherpa/onnx/cuda.html; \
    fi

COPY . .

EXPOSE 8000

CMD ["python", "run_litserve.py"]
