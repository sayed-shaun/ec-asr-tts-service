from typing import Literal, Union

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, overridable via env vars or .env.

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
    - GATEWAY_PORT is the only network setting left configurable: the
      gateway (pure FastAPI, no model loaded in-process; always binds
      0.0.0.0, not configurable — see main.py) is the one thing whose port
      genuinely varies per deployment. LitServe's own port (fixed at 8000 —
      see run_litserve.py) and where the gateway reaches it ("litserver:8000",
      see src/api/client.py's LITSERVE_BASE_URL) are hardcoded, not
      Settings fields: this assumes Docker Compose's network (the
      "litserver" hostname resolves via Docker's internal DNS), so bare
      two-terminal local dev without Compose needs a hosts-file entry
      mapping "litserver" to 127.0.0.1, or an equivalent override, to keep
      working. LitServe's inference route ("/predict") isn't a Settings
      field either — it's LitServe's own built-in default, left unset in
      litserver/server.py rather than duplicated as a constant; src/api/client.py
      hardcodes the same literal for the gateway's outbound calls.
    - SPEAKER_GATE_ENABLED defaults off: it's a real feature (segment-level
      speaker verification for live-cc — enroll the first SPEAKER_ENROLL_SECONDS
      of a call, then drop chunks that don't match that voice closely enough)
      but SPEAKER_SIMILARITY_THRESHOLD=0.75 is a generic starting point from
      speaker-verification literature, not measured against this project's own
      calls. Tune it against real recordings before trusting it — too low lets
      background talkers through, too high clips the enrolled speaker's own
      quieter segments.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    MODEL_NAME: str = "hishab/titu_stt_bn_fastconformer"
    ACCELERATOR: Literal["cpu", "cuda"] = "cuda"
    DEVICES: Union[int, Literal["auto"]] = 1

    WORKERS_PER_DEVICE: int = 2
    TRANSCRIBE_BATCH_SIZE: int = 4
    SAMPLE_RATE: int = 16000
    DENOISE: bool = False
    DENOISE_STATIONARY: bool = True
    MAX_SEGMENT_SECONDS: float = 18.0

    GATEWAY_PORT: int = 8000

    LIVE_CC_CHUNK_SECONDS: float = 3.0
    LIVE_CC_INTERIM_INTERVAL_SECONDS: float = 1.0
    LIVE_CC_INPUT_SAMPLE_RATE: int = 8000

    SPEAKER_GATE_ENABLED: bool = False
    SPEAKER_ENROLL_SECONDS: float = 3.0
    SPEAKER_SIMILARITY_THRESHOLD: float = 0.75

    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = ".logs"


settings = Settings()
