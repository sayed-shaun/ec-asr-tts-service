import base64
import os
import tempfile

import librosa
import numpy as np


def decode_base64_audio(audio_content: str, target_sr: int) -> np.ndarray:
    """Decode a base64-encoded audio blob (wav/flac/ogg/mp3/webm/...) into a
    mono float32 waveform resampled to `target_sr`.
    """
    try:
        raw_bytes = base64.b64decode(audio_content, validate=True)
    except Exception as exc:
        raise ValueError(f"Invalid base64 audio content: {exc}") from exc

    if not raw_bytes:
        raise ValueError("Decoded audio content is empty")

    # librosa.load can't decode compressed containers (webm/opus, mp4, ...)
    # from an in-memory BytesIO — soundfile doesn't support them at all, and
    # its audioread/ffmpeg fallback needs a real file path to shell out to,
    # not a stream. Round-tripping through a temp file makes every format
    # ffmpeg supports work, not just what soundfile can read from a buffer.
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

    return waveform.astype(np.float32)
