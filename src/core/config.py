from typing import Literal, Union

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, overridable via ASR_*-prefixed env vars or .env.

    Notes on non-obvious defaults:
    - DEVICES is pinned rather than "auto" because LitServe's "auto" device
      count shells out to `nvidia-smi -L` (not torch.cuda); on a machine where
      nvidia-smi is broken (NVML driver/library mismatch) but
      torch.cuda.is_available() is genuinely True, "auto" silently resolves to
      0 devices and server.run() crashes.
    - WORKERS_PER_DEVICE: each worker holds a full model copy in GPU memory;
      2 is conservative for ~8-12GB GPUs.
    - DENOISE defaults off: measured against this pipeline's sample data, it
      rewrote the transcript on every file (word count rose on all 8, 0
      matched the undenoised text) with unverified net effect on accuracy (no
      ground truth WER available). Only enable after verifying with real WER.
    - DENOISE_STATIONARY defaults True: measured better than non-stationary on
      this pipeline's sample data, but non-stationary is usually correct for
      real recordings with time-varying background noise — re-measure for
      your own input before trusting the default.
    - MAX_SEGMENT_SECONDS=18.0: this checkpoint was trained on clips up to
      ~18.5s; longer audio is split into segments of this length to avoid the
      encoder's relative-attention memory blowing up on multi-minute inputs.
    - LIVE_CC_CHUNK_SECONDS is the final caption latency (hard cuts, no
      overlap). LIVE_CC_INTERIM_INTERVAL_SECONDS re-transcribes the whole
      in-progress chunk on each tick to fake incremental decoding — total GPU
      work per chunk scales roughly as chunk_seconds / this value; set to 0 to
      disable interim captions. LIVE_CC_INPUT_SAMPLE_RATE must match the raw
      PCM rate the client actually streams, not SAMPLE_RATE.
    - LITSERVE_HOST/LITSERVE_PORT is the address the gateway (HOST/API_PORT,
      pure FastAPI, no model loaded in-process) uses to reach the separate
      LitServe model server process/container (see run_litserve.py) over
      real HTTP. LitServe itself always binds 0.0.0.0 on its own container —
      this setting is only the gateway's *connect* target, e.g. "127.0.0.1"
      for two local processes on one host, or the service name (e.g.
      "litserver") when they're separate containers on a Docker network.
    """

    model_config = SettingsConfigDict(env_prefix="ASR_", env_file=".env", extra="ignore")

    MODEL_NAME: str = "hishab/titu_stt_bn_fastconformer"

    ACCELERATOR: Literal["cpu", "cuda"] = "cuda"

    DEVICES: Union[int, Literal["auto"]] = 1

    WORKERS_PER_DEVICE: int = 2

    TRANSCRIBE_BATCH_SIZE: int = 4

    SAMPLE_RATE: int = 16000

    DENOISE: bool = False

    DENOISE_STATIONARY: bool = True

    MAX_SEGMENT_SECONDS: float = 18.0

    HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_PATH: str = "/api/v1/asr/transcribe"

    LITSERVE_HOST: str = "127.0.0.1"
    LITSERVE_PORT: int = 8001

    LIVE_CC_CHUNK_SECONDS: float = 3.0

    LIVE_CC_INTERIM_INTERVAL_SECONDS: float = 1.0

    LIVE_CC_INPUT_SAMPLE_RATE: int = 8000

    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = ".logs"


settings = Settings()
