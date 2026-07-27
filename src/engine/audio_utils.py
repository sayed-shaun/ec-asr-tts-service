import base64
import io

import librosa
import numpy as np


def decode_base64_audio(audio_content: str, target_sr: int) -> np.ndarray:
    """Decode a base64-encoded audio blob (wav/flac/ogg/mp3/...) into a mono
    float32 waveform resampled to `target_sr`.
    """
    try:
        raw_bytes = base64.b64decode(audio_content, validate=True)
    except Exception as exc:
        raise ValueError(f"Invalid base64 audio content: {exc}") from exc

    if not raw_bytes:
        raise ValueError("Decoded audio content is empty")

    try:
        waveform, _ = librosa.load(io.BytesIO(raw_bytes), sr=target_sr, mono=True)
    except Exception as exc:
        raise ValueError(f"Unable to decode audio: {exc}") from exc

    if waveform.size == 0:
        raise ValueError("Decoded waveform is empty")

    return waveform.astype(np.float32)
