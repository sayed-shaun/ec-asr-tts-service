from typing import Literal, Union

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, overridable via env vars or .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    MODEL_NAME: str = "hishab/titu_stt_bn_conformer_large"
    ACCELERATOR: Literal["cpu", "cuda"] = "cuda"
    # Pinned rather than "auto": LitServe's "auto" shells out to `nvidia-smi -L`,
    # which can silently report 0 devices (NVML mismatch) even when
    # torch.cuda.is_available() is True, crashing server.run().
    DEVICES: Union[int, Literal["auto"]] = 1

    # Each worker holds a full model copy in GPU memory; 2 is conservative
    # for ~8-12GB GPUs.
    WORKERS_PER_DEVICE: int = 2
    TRANSCRIBE_BATCH_SIZE: int = 4
    SAMPLE_RATE: int = 16000
    # Off by default: measured on this pipeline's sample data it rewrote every
    # transcript with unverified WER impact. Only enable after measuring.
    DENOISE: bool = False
    # True measured better on sample data, but non-stationary noise reduction
    # usually suits real recordings better — re-measure before trusting this.
    DENOISE_STATIONARY: bool = True
    # This checkpoint trained on clips up to ~18.5s; longer audio is segmented
    # to avoid blowing up the encoder's relative-attention memory.
    MAX_SEGMENT_SECONDS: float = 18.0

    # The only configurable network setting — LitServe's host/port and
    # /predict route are hardcoded elsewhere (run_litserve.py, src/api/client.py).
    GATEWAY_PORT: int = 8000

    # Final caption latency (hard cuts, no overlap).
    LIVE_CC_CHUNK_SECONDS: float = 3.0
    # Re-transcribes the in-progress chunk each tick to fake incremental
    # decoding; GPU cost scales as chunk_seconds / this value. 0 disables
    # interim captions.
    LIVE_CC_INTERIM_INTERVAL_SECONDS: float = 1.0
    # Must match the raw PCM rate the client actually streams, not SAMPLE_RATE.
    LIVE_CC_INPUT_SAMPLE_RATE: int = 8000

    SPEAKER_GATE_ENABLED: bool = False
    SPEAKER_ENROLL_SECONDS: float = 3.0
    # Generic speaker-verification starting point, not measured against this
    # project's own calls — tune against real recordings before trusting it.
    SPEAKER_SIMILARITY_THRESHOLD: float = 0.75

    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = ".logs"


settings = Settings()
