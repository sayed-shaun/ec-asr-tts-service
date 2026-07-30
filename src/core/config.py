from typing import Literal, Union

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ASR_", env_file=".env", extra="ignore")

    MODEL_NAME: str = "hishab/titu_stt_bn_fastconformer"

    # Human-readable label for GET /v1/asr/info. Not derived from MODEL_NAME
    # automatically: that route runs in the main process and must not touch
    # the engine (the loaded model lives in separate LitServe worker
    # processes — see router.py), so it can't introspect the model's real
    # class. Set this alongside MODEL_NAME when you change it, or /v1/asr/info
    # reports the previous model's architecture.
    MODEL_ARCHITECTURE: str = "FastConformer-CTC (NeMo)"

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

    # Spectral-gating denoise (noisereduce) applied to every decoded waveform
    # before transcription. Off by default: measured against this pipeline's
    # own sample data, it rewrote the transcript on every file (word count rose
    # on all 8, 0 produced text identical to undenoised) and only improved an
    # invalid-Bengali-orthography proxy in aggregate (79->69), regressing on
    # 2 of 8 files individually. Whether the added words are recovered speech
    # or denoise artifacts is unverified — no ground truth to score against.
    # Only enable if your real input is noisier than the sample data and you
    # can verify with actual WER, not this proxy.
    DENOISE: bool = False

    # stationary=True measured better than False (non-stationary) on this
    # pipeline's sample data (79->69 vs 79->84 on the same orthography proxy).
    # Non-stationary re-estimates the noise floor continuously, which is
    # usually the right choice for real recordings with time-varying
    # background noise — re-measure before trusting the stationary default
    # if your input doesn't resemble clean-ish broadcast narration.
    DENOISE_STATIONARY: bool = True

    # This checkpoint was trained on clips up to ~18.5s; longer audio is split
    # into segments of this length before transcription to avoid the encoder's
    # relative-attention memory blowing up on multi-minute inputs.
    MAX_SEGMENT_SECONDS: float = 18.0

    HOST: str = "0.0.0.0"
    PORT: int = 8000
    API_PATH: str = "/api/v1/asr/transcribe"

    # live-cc buffers raw PCM off the WebSocket and transcribes one chunk at a
    # time — this is the FINAL caption latency (and, since chunks are hard
    # cuts with no overlap, the rough granularity words can get split at).
    LIVE_CC_CHUNK_SECONDS: float = 3.0

    # Every this many seconds of in-progress audio, re-transcribe the whole
    # in-progress chunk so far and push it as an interim (is_final: false)
    # caption — gives the feel of live-updating text using the same offline
    # model as a black box, no incremental decoding. Real cost: each interim
    # re-transcribes from the start of the current chunk, so total GPU work
    # per chunk scales up (roughly chunk_seconds / this, redundant compute).
    # Set to 0 to disable interim updates and only emit final captions.
    LIVE_CC_INTERIM_INTERVAL_SECONDS: float = 1.0

    # The sample rate of the raw PCM the client actually streams over the
    # WebSocket — used both to label the WAV header correctly (so librosa
    # resamples from the true source rate to SAMPLE_RATE, not a mislabeled
    # one) and to compute how many bytes make up one LIVE_CC_CHUNK_SECONDS
    # buffer. Set this to match your actual input, not SAMPLE_RATE.
    LIVE_CC_INPUT_SAMPLE_RATE: int = 8000

    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = ".logs"


settings = Settings()
