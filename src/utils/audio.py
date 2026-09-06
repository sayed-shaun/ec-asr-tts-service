import base64
import io
import os
import subprocess
import tempfile
import wave

import librosa
import numpy as np


def ffmpeg_decode(path: str, target_sr: int) -> np.ndarray:
    """Decode any container ffmpeg understands into mono float32 at target_sr.

    The fallback for formats libsndfile cannot open -- webm/opus above all,
    which is what browser MediaRecorder produces. librosa used to reach
    audioread for this, but 1.0 dropped that fallback, so shelling out is now
    the only path that works across librosa versions.
    """
    result = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-loglevel", "error", "-i", path,
            "-f", "f32le", "-ac", "1", "-ar", str(target_sr), "pipe:1",
        ],
        capture_output=True,
    )
    if result.returncode != 0 or not result.stdout:
        detail = result.stderr.decode("utf-8", "replace").strip()[:200]
        raise ValueError(f"ffmpeg could not decode audio: {detail}")
    return np.frombuffer(result.stdout, dtype=np.float32)


def decode_base64_audio(audio_content: str, target_sr: int) -> np.ndarray:
    """Decode a base64-encoded audio blob into a mono float32 waveform
    resampled to `target_sr`.

    Round-trips through a temp file because ffmpeg needs a real path. libsndfile
    handles wav/flac/ogg directly; anything else -- webm/opus from a browser,
    mp4, mp3 -- goes through ffmpeg_decode above.
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
        try:
            waveform, _ = librosa.load(tmp_path, sr=target_sr, mono=True)
        except Exception:
            waveform = ffmpeg_decode(tmp_path, target_sr)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Unable to decode audio: {exc}") from exc
    finally:
        if tmp_path is not None:
            os.unlink(tmp_path)

    if waveform.size == 0:
        raise ValueError("Decoded waveform is empty")

    return waveform.astype(np.float32)


def warm_audio_decoder(target_sr: int) -> None:
    """Force librosa/soundfile/soxr's lazy initialization at startup.

    The first decode in a process costs ~920ms in lazy imports and codec setup
    vs ~0.6ms afterwards. The frame rate is target_sr // 2, not target_sr, so
    the resampler's own one-off init is exercised too.
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


def wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Frame float32 mono samples as a 16-bit PCM WAV file."""
    clipped = np.clip(samples, -1.0, 1.0)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes((clipped * 32767.0).astype("<i2", copy=False).tobytes())
    return buf.getvalue()
