import base64
import io
import os
import tempfile
import wave

import librosa
import numpy as np

from src.core.config import settings


def decode_base64_audio(audio_content: str, target_sr: int) -> np.ndarray:
    """Decode a base64-encoded audio blob (wav/flac/ogg/mp3/webm/...) into a
    mono float32 waveform resampled to `target_sr`.

    Round-trips through a temp file rather than an in-memory buffer:
    librosa.load can't decode compressed containers (webm/opus, mp4, ...) from
    an in-memory BytesIO, soundfile doesn't support them at all, and its
    audioread/ffmpeg fallback needs a real file path to shell out to, not a
    stream. A temp file makes every format ffmpeg supports work.
    """
    try:
        raw_bytes = base64.b64decode(audio_content, validate=True)
    except Exception as exc:
        raise ValueError(f"Invalid base64 audio content: {exc}") from exc

    if not raw_bytes:
        raise ValueError("Decoded audio content is empty")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(raw_bytes)
            tmp_path = tmp_file.name
        waveform, _ = librosa.load(tmp_path, sr=target_sr, mono=True)
    except Exception as exc:
        raise ValueError(f"Unable to decode audio: {exc}") from exc
    finally:
        if tmp_path is not None:
            os.unlink(tmp_path)

    if waveform.size == 0:
        raise ValueError("Decoded waveform is empty")

    waveform = waveform.astype(np.float32)
    if settings.DENOISE:
        waveform = denoise(waveform, target_sr)
    return waveform


def denoise(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """Spectral-gating noise reduction (see DENOISE in config.py for the
    measured tradeoffs before enabling this).
    """
    import noisereduce as nr

    reduced = nr.reduce_noise(
        y=waveform, sr=sample_rate, stationary=settings.DENOISE_STATIONARY
    )
    return reduced.astype(np.float32)


def warm_audio_decoder(target_sr: int) -> None:
    """Force librosa/soundfile/soxr's lazy initialization at startup.

    The first decode_base64_audio() call in a process costs ~920ms (measured)
    purely in lazy imports and codec/resampler setup, vs ~0.6ms afterwards.
    Without this the first real request eats that cost. Best-effort: a failed
    warmup must not stop a worker from coming up.

    Uses target_sr // 2 as the WAV frame rate (deliberately not target_sr) to
    also exercise the resampler, which has its own one-off init cost on the
    first non-passthrough conversion.
    """
    samples = np.zeros(target_sr // 10, dtype=np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(target_sr // 2)
        wav_file.writeframes(samples.tobytes())

    decode_base64_audio(
        base64.b64encode(buf.getvalue()).decode("utf-8"), target_sr=target_sr
    )
