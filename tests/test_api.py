import base64
import io
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.api.v1.asr.router import router as asr_router
from src.api.v1.asr.schema import AsrRequest
from src.api.v1.live_cc.router import router as live_cc_router
from src.core.config import settings
from src.models.conformer.engine import ASREngine
from src.models.conformer.litapi import ASRLitAPI
from src.utils.audio import decode_base64_audio


def _make_wav_base64(seconds: float = 0.5, sr: int = 16000) -> str:
    samples = (np.sin(2 * np.pi * 440 * np.arange(int(sr * seconds)) / sr) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@pytest.fixture
def info_client():
    app = FastAPI()
    app.include_router(asr_router)
    return TestClient(app)


def test_info_endpoint(info_client):
    resp = info_client.get("/v1/asr/info")
    assert resp.status_code == 200
    assert resp.json()["model"] == "hishab/titu_stt_bn_fastconformer"


def test_decode_base64_audio_roundtrip():
    b64 = _make_wav_base64(seconds=1.0, sr=16000)
    waveform = decode_base64_audio(b64, target_sr=16000)
    assert waveform.dtype == np.float32
    assert 15900 <= waveform.shape[0] <= 16100


def test_decode_base64_audio_resamples():
    b64 = _make_wav_base64(seconds=1.0, sr=8000)
    waveform = decode_base64_audio(b64, target_sr=16000)
    assert 15900 <= waveform.shape[0] <= 16100


def test_decode_base64_audio_rejects_garbage():
    with pytest.raises(ValueError):
        decode_base64_audio(base64.b64encode(b"not audio").decode(), target_sr=16000)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="requires ffmpeg")
def test_decode_base64_audio_handles_webm_opus_and_cleans_up_temp_file():
    # Regression test: librosa can't decode compressed containers (webm/opus,
    # what MediaRecorder produces in browsers) from an in-memory BytesIO —
    # soundfile doesn't support the format at all, and its audioread/ffmpeg
    # fallback needs a real file path, not a stream.
    with tempfile.TemporaryDirectory() as tmp_dir:
        webm_path = Path(tmp_dir) / "test.webm"
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=0.5", "-c:a", "libopus", str(webm_path)],
            check=True,
            capture_output=True,
        )
        b64 = base64.b64encode(webm_path.read_bytes()).decode()

    before = set(Path(tempfile.gettempdir()).iterdir())
    waveform = decode_base64_audio(b64, target_sr=16000)
    after = set(Path(tempfile.gettempdir()).iterdir())

    assert waveform.dtype == np.float32
    assert waveform.size > 0
    assert after == before  # no leftover temp file


def test_asr_request_rejects_empty_audio_list():
    with pytest.raises(ValidationError):
        AsrRequest(config={}, audio=[])


def test_asr_request_defaults():
    req = AsrRequest(audio=[{"audioContent": "abc"}])
    assert req.config.language.sourceLanguage == "bn"


@pytest.fixture
def lit_api():
    api = ASRLitAPI(max_batch_size=1, api_path=settings.API_PATH)
    api.engine = MagicMock()
    api.engine.model = object()
    api.engine.transcribe.return_value = ["হ্যালো"]
    return api


def test_lit_api_full_cycle(lit_api):
    request = AsrRequest(audio=[{"audioContent": _make_wav_base64()}])
    decoded = lit_api.decode_request(request)
    assert len(decoded["audios"]) == 1

    prediction = lit_api.predict(decoded)
    lit_api.engine.transcribe.assert_called_once()

    response = lit_api.encode_response(prediction)
    assert response.taskType == "asr"
    assert response.output[0].source == "হ্যালো"
    assert response.time_taken >= 0


def test_asr_engine_splits_long_audio_into_segments():
    engine = ASREngine(model_name="dummy", device="cpu", max_segment_seconds=1.0)
    engine.model = MagicMock()
    engine.model.transcribe.return_value = ["a", "b", "c"]

    sr = 16000
    audio = np.zeros(int(sr * 2.5), dtype=np.float32)
    result = engine.transcribe([audio], batch_size=4, sample_rate=sr)

    assert result == ["a b c"]
    assert len(engine.model.transcribe.call_args.kwargs["audio"]) == 3


def test_asr_engine_leaves_short_audio_unsplit():
    engine = ASREngine(model_name="dummy", device="cpu", max_segment_seconds=18.0)
    engine.model = MagicMock()
    engine.model.transcribe.return_value = ["hello"]

    sr = 16000
    audio = np.zeros(int(sr * 2.0), dtype=np.float32)
    result = engine.transcribe([audio], batch_size=4, sample_rate=sr)

    assert result == ["hello"]
    assert len(engine.model.transcribe.call_args.kwargs["audio"]) == 1


def test_asr_engine_respects_custom_sample_rate_for_segment_length():
    engine = ASREngine(model_name="dummy", device="cpu", max_segment_seconds=1.0)
    engine.model = MagicMock()
    engine.model.transcribe.return_value = ["a", "b"]

    audio = np.zeros(8000 * 2, dtype=np.float32)
    result = engine.transcribe([audio], batch_size=4, sample_rate=8000)

    assert result == ["a b"]
    segments = engine.model.transcribe.call_args.kwargs["audio"]
    assert len(segments) == 2
    assert all(len(seg) <= 8000 for seg in segments)


@pytest.fixture
def live_cc_client():
    app = FastAPI()
    app.include_router(live_cc_router)

    @app.post(settings.API_PATH)
    async def fake_predict(payload: dict) -> dict:
        return {"taskType": "asr", "output": [{"source": "হ্যালো"}], "time_taken": 0.1}

    return TestClient(app)


def test_live_cc_ws_emits_caption_per_chunk(live_cc_client):
    chunk_samples = int(settings.LIVE_CC_INPUT_SAMPLE_RATE * settings.LIVE_CC_CHUNK_SECONDS)
    pcm = np.zeros(chunk_samples, dtype=np.int16).tobytes()

    with live_cc_client.websocket_connect("/v1/live-cc/ws") as ws:
        ws.send_bytes(pcm)
        message = ws.receive_json()

    assert message == {"text": "হ্যালো", "is_final": True}


def test_live_cc_ws_buffers_partial_chunks(live_cc_client):
    # Half a chunk (1.5s) already crosses one interim step (1.0s), so the first
    # half triggers an interim caption before the second half finalizes it.
    chunk_samples = int(settings.LIVE_CC_INPUT_SAMPLE_RATE * settings.LIVE_CC_CHUNK_SECONDS)
    half_pcm = np.zeros(chunk_samples // 2, dtype=np.int16).tobytes()

    with live_cc_client.websocket_connect("/v1/live-cc/ws") as ws:
        ws.send_bytes(half_pcm)
        interim = ws.receive_json()
        ws.send_bytes(half_pcm)
        final = ws.receive_json()

    assert interim == {"text": "হ্যালো", "is_final": False}
    assert final == {"text": "হ্যালো", "is_final": True}


def test_live_cc_ws_emits_interim_captions_before_final(live_cc_client):
    interim_samples = int(settings.LIVE_CC_INPUT_SAMPLE_RATE * settings.LIVE_CC_INTERIM_INTERVAL_SECONDS)
    step_pcm = np.zeros(interim_samples, dtype=np.int16).tobytes()

    chunk_seconds = settings.LIVE_CC_CHUNK_SECONDS
    interim_seconds = settings.LIVE_CC_INTERIM_INTERVAL_SECONDS
    num_interim_steps = int(chunk_seconds / interim_seconds) - 1

    with live_cc_client.websocket_connect("/v1/live-cc/ws") as ws:
        messages = []
        for _ in range(num_interim_steps):
            ws.send_bytes(step_pcm)
            messages.append(ws.receive_json())

        # Send the remainder to cross the final chunk boundary.
        remaining_samples = int(settings.LIVE_CC_INPUT_SAMPLE_RATE * chunk_seconds) - interim_samples * num_interim_steps
        ws.send_bytes(np.zeros(remaining_samples, dtype=np.int16).tobytes())
        messages.append(ws.receive_json())

    assert all(m["is_final"] is False for m in messages[:-1])
    assert messages[-1]["is_final"] is True
