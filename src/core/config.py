from typing import Literal, Union

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ASR_", env_file=".env", extra="ignore")

    MODEL_NAME: str = "hishab/titu_stt_bn_fastconformer"
    ACCELERATOR: Literal["cpu", "cuda"] = "cuda"

    # LitServe's "auto" device count shells out to `nvidia-smi -L` (not
    # torch.cuda) — on a machine where nvidia-smi is broken (e.g. an NVML
    # driver/library version mismatch) but torch.cuda.is_available() is
    # genuinely True, "auto" silently resolves to 0 devices and server.run()
    # crashes with "num_api_servers must be greater than 0". Pin the real
    # GPU count explicitly instead of depending on nvidia-smi being healthy.
    DEVICES: Union[int, Literal["auto"]] = 1

    # Each worker is a separate process holding its own full copy of the model
    # in GPU memory — this is the main throughput lever for concurrent requests,
    # but memory cost multiplies per worker. 2 is a conservative default for
    # ~8-12GB GPUs; raise it if `nvidia-smi` shows headroom, lower it on OOM.
    WORKERS_PER_DEVICE: int = 2

    # Batch size for NeMo's own ASRModel.transcribe() call, applied across the
    # audio items *within* a single request (not across concurrent requests).
    TRANSCRIBE_BATCH_SIZE: int = 4

    SAMPLE_RATE: int = 16000

    # This checkpoint was trained on clips up to ~18.5s; longer audio is split
    # into segments of this length before transcription to avoid the encoder's
    # relative-attention memory blowing up on multi-minute inputs.
    MAX_SEGMENT_SECONDS: float = 18.0

    HOST: str = "0.0.0.0"
    PORT: int = 8000
    API_PATH: str = "/api/v1/asr/transcribe"

    # live-cc buffers raw PCM off the WebSocket and transcribes one chunk at a
    # time — this is the caption latency (and, since chunks are hard cuts with
    # no overlap, the rough granularity words can get split at).
    LIVE_CC_CHUNK_SECONDS: float = 3.0

    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = ".logs"


settings = Settings()
