from typing import Literal, Union

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ASR_", env_file=".env", extra="ignore")

    model_name: str = "hishab/titu_stt_bn_conformer_large"
    accelerator: Literal["cpu", "cuda", "mps", "auto"] = "auto"
    devices: Union[int, Literal["auto"]] = "auto"
    workers_per_device: int = 1

    # Batch size for NeMo's own ASRModel.transcribe() call, applied across the
    # audio items *within* a single request (not across concurrent requests).
    transcribe_batch_size: int = 4

    sample_rate: int = 16000

    # This checkpoint was trained on clips up to ~18.5s; longer audio is split
    # into segments of this length before transcription to avoid the encoder's
    # relative-attention memory blowing up on multi-minute inputs.
    max_segment_seconds: float = 18.0

    host: str = "0.0.0.0"
    port: int = 8000
    api_path: str = "/predict"

    log_level: str = "INFO"
    log_dir: str = "log_folder"


settings = Settings()
