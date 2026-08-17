from typing import Literal, Union

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, overridable via env vars or .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ENGINE: Literal["conformer", "wav2vec2", "whisper", "zipformer"] = "conformer"
    CONFORMER_MODEL_NAME: str = "bengaliAI/BanglaConformer"
    WAV2VEC2_MODEL_NAME: str = "SayedShaun/bangla-wave2vec2-unigram"
    WHISPER_MODEL_NAME: str = "bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium"
    ZIPFORMER_MODEL_NAME: str = "alphacep/vosk-model-small-streaming-bn"
    ACCELERATOR: Literal["cpu", "cuda"] = "cuda"
    DEVICES: Union[int, Literal["auto"]] = 1

    WORKERS_PER_DEVICE: int = 1
    TRANSCRIBE_BATCH_SIZE: int = 4
    SAMPLE_RATE: int = 16000
    MAX_SEGMENT_SECONDS: float = 8.0

    LITSERVE_TIMEOUT: float = 120.0

    ITN_ENABLED: bool = True

    GATEWAY_PORT: int = 8000

    LIVE_CC_CHUNK_SECONDS: float = 3.0
    LIVE_CC_INTERIM_INTERVAL_SECONDS: float = 1.0
    LIVE_CC_INPUT_SAMPLE_RATE: int = 16000

    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = ".logs"

    @property
    def ACTIVE_MODEL_NAME(self) -> str:
        """Whichever checkpoint settings.ENGINE will actually load."""
        if self.ENGINE == "wav2vec2":
            return self.WAV2VEC2_MODEL_NAME
        if self.ENGINE == "whisper":
            return self.WHISPER_MODEL_NAME
        if self.ENGINE == "zipformer":
            return self.ZIPFORMER_MODEL_NAME
        if self.ENGINE == "conformer":
            return self.CONFORMER_MODEL_NAME
        raise ValueError(
            f"Unknown engine: {self.ENGINE}", 
            "must be one of conformer, wav2vec2, whisper, zipformer"
        )


settings = Settings()
