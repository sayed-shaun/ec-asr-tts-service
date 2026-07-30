FROM pytorch/pytorch:2.9.1-cuda13.0-cudnn9-runtime

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

# This base image already has torch + CUDA 13 + cuDNN 9 installed — pip sees
# our own "torch>=2.1.0" is already satisfied and skips it entirely, so this
# only downloads nemo_toolkit and everything else. That's most of the weight
# a from-scratch `pip install torch nemo_toolkit[asr]` would otherwise pull.
COPY pyproject.toml .
COPY src/ src/
# ".[denoise]" so ASR_DENOISE can be toggled at runtime (env var) without a
# rebuild — noisereduce (numpy/scipy-based, no extra ML weight) ships either way.
RUN pip install --no-cache-dir ".[denoise]"

COPY . .

EXPOSE 8000

CMD ["python", "main.py"]
