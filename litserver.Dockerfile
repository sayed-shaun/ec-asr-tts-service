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

# The PyPI sherpa-onnx installed above is CPU-only, so replace it with the CUDA
# wheel that ZIPFORMER_PROVIDER=cuda needs. This pin is the only statement of
# which wheel the image carries, and must match the base image's CUDA/cuDNN
# pair (cuda12.8/cudnn9 -> the cuda12.cudnn9 variant) -- picked from
# https://k2-fsa.github.io/sherpa/onnx/cuda.html. The CUDA wheel does not
# bundle cuDNN: it needs libcudnn.so.9 from the system, which this base
# supplies and a slimmer one would not.
#
# It is a build arg only so a CPU-only box can pass --build-arg to skip it
# (alongside ZIPFORMER_PROVIDER=cpu: the wheel supplies the provider, the
# setting selects it). Nothing in .env or compose overrides it.
ARG SHERPA_ONNX_CUDA_VERSION=1.13.5+cuda12.cudnn9.onnxruntime1.27.1
RUN if [ "$SHERPA_ONNX_CUDA_VERSION" != "cpu" ]; then \
        pip install --no-cache-dir --break-system-packages --force-reinstall \
            "sherpa-onnx==${SHERPA_ONNX_CUDA_VERSION}" \
            -f https://k2-fsa.github.io/sherpa/onnx/cuda.html; \
    fi

# Put the CUDA libs on the loader path. This base ships CUDA not as system
# packages but as pip nvidia-* wheels under site-packages: torch resolves those
# itself, so TTS works without help, but onnxruntime's CUDA provider is opened
# with plain dlopen and finds nothing. Without this, loading the recognizer
# dies on "libcublasLt.so.12: cannot open shared object file" -- and note it
# raises rather than falling back to CPU, so ZIPFORMER_PROVIDER=cuda would take
# the whole ASR worker down at startup.
#
# ldconfig rather than ENV LD_LIBRARY_PATH so it cannot be clobbered by a
# caller passing their own. This is also what makes ASR and TTS share one CUDA:
# both end up on these same cublas/cudnn 12.x/9 libs.
RUN printf '%s\n' /usr/local/lib/python3.12/dist-packages/nvidia/*/lib \
        > /etc/ld.so.conf.d/nvidia-pip.conf \
    && ldconfig

COPY . .

EXPOSE 8000

CMD ["python", "run_litserve.py"]
