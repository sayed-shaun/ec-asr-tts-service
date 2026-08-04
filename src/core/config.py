from typing import Literal, Union

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, overridable via env vars or .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    MODEL_NAME: str = "hishab/titu_stt_bn_conformer_large"
    ACCELERATOR: Literal["cpu", "cuda"] = "cuda"
    DEVICES: Union[int, Literal["auto"]] = 1

    WORKERS_PER_DEVICE: int = 2
    TRANSCRIBE_BATCH_SIZE: int = 4
    SAMPLE_RATE: int = 16000
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
