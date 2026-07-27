from typing import Literal, Union

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ASR_", env_file=".env", extra="ignore")

    MODEL_NAME: str = "hishab/titu_stt_bn_fastconformer"
    ACCELERATOR: Literal["cpu", "cuda"] = "cuda"
    DEVICES: Union[int, Literal["auto"]] = "auto"

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

    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = ".logs"


settings = Settings()
