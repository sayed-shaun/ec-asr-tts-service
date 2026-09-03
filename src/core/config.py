from typing import Literal, Union

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, overridable via env vars or .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ZIPFORMER_MODEL_NAME: str = "alphacep/vosk-model-small-streaming-bn"
    ZIPFORMER_PROVIDER: Literal["cpu", "cuda"] = "cpu"

    TTS_ENABLED: bool = False
    TTS_MODEL_NAME: str = "ai4bharat/indic-parler-tts"
    TTS_VOICE: str = "Aditi"
    TTS_MAX_CHARS: int = 160

    ACCELERATOR: Literal["cpu", "cuda"] = "cuda"
    DEVICES: Union[int, Literal["auto"]] = 1

    WORKERS_PER_DEVICE: int = 1
    TRANSCRIBE_BATCH_SIZE: int = 4
    SAMPLE_RATE: int = 16000

    LITSERVE_TIMEOUT: float = 120.0

    ITN_ENABLED: bool = True

    GATEWAY_PORT: int = 8000
    LITSERVE_BASE_URL: str = "http://litserver:8000"
    LITSERVE_PORT: int = 8000
    CORS_ALLOW_ORIGINS: str = "*"

    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = ".logs"

    @property
    def ACTIVE_MODEL_NAME(self) -> str:
        """The ASR checkpoint the server will load."""
        return self.ZIPFORMER_MODEL_NAME

    @property
    def ACTIVE_TTS_MODEL_NAME(self) -> str | None:
        """Whichever TTS checkpoint the server will load, None when disabled."""
        return self.TTS_MODEL_NAME if self.TTS_ENABLED else None

settings = Settings()
