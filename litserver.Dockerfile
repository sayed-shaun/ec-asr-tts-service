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

COPY pyproject.toml .
RUN mkdir -p src && touch src/__init__.py
RUN pip install --no-cache-dir --break-system-packages ".[serve]"

ARG SHERPA_ONNX_CUDA_VERSION=1.13.5+cuda12.cudnn9.onnxruntime1.27.1
RUN if [ "$SHERPA_ONNX_CUDA_VERSION" != "cpu" ]; then \
        pip install --no-cache-dir --break-system-packages --force-reinstall \
            "sherpa-onnx==${SHERPA_ONNX_CUDA_VERSION}" \
            -f https://k2-fsa.github.io/sherpa/onnx/cuda.html; \
    fi

RUN printf '%s\n' /usr/local/lib/python3.12/dist-packages/nvidia/*/lib \
        > /etc/ld.so.conf.d/nvidia-pip.conf \
    && ldconfig

COPY . .

EXPOSE 8000

CMD ["python", "run_litserve.py"]
