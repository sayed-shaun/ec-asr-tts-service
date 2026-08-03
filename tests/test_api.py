import base64
import io
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from main import create_gateway_app
from src.api.client import PREDICT_PATH
from src.api.v1.asr.router import router as asr_router
from src.api.v1.asr.schema import AsrRequest
from src.api.v1.live_cc.router import router as live_cc_router
from src.core.config import settings
from src.litserver.engine import ASREngine
from src.litserver.litapi import ASRLitAPI
from src.utils.audio import decode_base64_audio, denoise


def make_wav_base64(seconds: float = 0.5, sr: int = 16000) -> str:
    samples = (
        np.sin(2 * np.pi * 440 * np.arange(int(sr * seconds)) / sr) * 32767
    ).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@pytest.fixture
def info_client():
    return TestClient(create_gateway_app())


def test_info_endpoint(info_client):
    resp = info_client.get("/api/v1/asr/info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == settings.MODEL_NAME


@pytest.fixture
def asr_client(monkeypatch):
    """asr_router forwards to LitServe over a real HTTP client (see
    src/api/client.py); the fake predict endpoint below lives on a separate
    app, and the client's httpx.AsyncClient is monkeypatched to reach it
    in-process instead of opening a real socket, keeping the test hermetic.
    """
    predict_app = FastAPI()

    @predict_app.post(PREDICT_PATH)
    async def fake_predict(payload: dict) -> dict:
        return {
            "taskType": "asr",
            "output": [{"source": "হ্যালো"}],
            "time_taken": 0.1,
        }

    real_async_client = httpx.AsyncClient

    def fake_async_client(*args, **kwargs):
        return real_async_client(
            transport=httpx.ASGITransport(app=predict_app),
            base_url="http://internal",
        )

    monkeypatch.setattr("src.api.client.httpx.AsyncClient", fake_async_client)

    app = FastAPI()
    app.include_router(asr_router)
    return TestClient(app)


def test_transcribe_json_endpoint(asr_client):
    resp = asr_client.post(
        "/api/v1/asr/transcribe",
        json={"audio": [{"audioContent": make_wav_base64()}]},
    )
    assert resp.status_code == 200
    assert resp.json()["output"][0]["source"] == "হ্যালো"


def test_decode_base64_audio_roundtrip():
    b64 = make_wav_base64(seconds=1.0, sr=16000)
    waveform = decode_base64_audio(b64, target_sr=16000)
    assert waveform.dtype == np.float32
    assert 15900 <= waveform.shape[0] <= 16100


def test_decode_base64_audio_resamples():
    b64 = make_wav_base64(seconds=1.0, sr=8000)
    waveform = decode_base64_audio(b64, target_sr=16000)
    assert 15900 <= waveform.shape[0] <= 16100


def test_decode_base64_audio_rejects_garbage():
    with pytest.raises(ValueError):
        decode_base64_audio(base64.b64encode(b"not audio").decode(), target_sr=16000)


def test_decode_base64_audio_skips_denoise_by_default(monkeypatch):
    """DENOISE defaults to False — measured to rewrite the transcript on
    every file tested (word count rose on all of them, none matched the
    undenoised text), so it must stay opt-in, not silently applied.
    """
    called = []
    monkeypatch.setattr("src.utils.audio.denoise", lambda w, sr: called.append(1) or w)
    decode_base64_audio(make_wav_base64(), target_sr=16000)
    assert not called


def test_decode_base64_audio_denoises_when_enabled(monkeypatch):
    monkeypatch.setattr(settings, "DENOISE", True)
    called = []
    monkeypatch.setattr("src.utils.audio.denoise", lambda w, sr: called.append(1) or w)
    decode_base64_audio(make_wav_base64(), target_sr=16000)
    assert called


def test_denoise_preserves_dtype_and_length():
    waveform = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32)
    out = denoise(waveform, 16000)
    assert out.dtype == np.float32
    assert out.shape == waveform.shape


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="requires ffmpeg")
def test_decode_base64_audio_handles_webm_opus_and_cleans_up_temp_file():
    """Regression test: librosa can't decode compressed containers (webm/opus,
    what MediaRecorder produces in browsers) from an in-memory BytesIO —
    soundfile doesn't support the format at all, and its audioread/ffmpeg
    fallback needs a real file path, not a stream.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        webm_path = Path(tmp_dir) / "test.webm"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=0.5",
                "-c:a",
                "libopus",
                str(webm_path),
            ],
            check=True,
            capture_output=True,
        )
        b64 = base64.b64encode(webm_path.read_bytes()).decode()

    before = set(Path(tempfile.gettempdir()).iterdir())
    waveform = decode_base64_audio(b64, target_sr=16000)
    after = set(Path(tempfile.gettempdir()).iterdir())

    assert waveform.dtype == np.float32
    assert waveform.size > 0
    assert after == before


def test_asr_request_rejects_empty_audio_list():
    with pytest.raises(ValidationError):
        AsrRequest(config={}, audio=[])


def test_asr_request_defaults():
    req = AsrRequest(audio=[{"audioContent": "abc"}])
    assert req.config.language.sourceLanguage == "bn"


@pytest.fixture
def lit_api():
    api = ASRLitAPI(max_batch_size=1, api_path=PREDICT_PATH)
    api.engine = MagicMock()
    api.engine.model = object()
    api.engine.transcribe.return_value = ["হ্যালো"]
    return api


def test_lit_api_full_cycle(lit_api):
    request = AsrRequest(audio=[{"audioContent": make_wav_base64()}])
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
def live_cc_client(monkeypatch):
    """live_cc_router now calls the LitServe model server over a real HTTP
    client (see router.py) rather than an in-process ASGI transport, so the
    fake predict endpoint below lives on a separate app; the router's
    httpx.AsyncClient is monkeypatched to reach it in-process instead of
    opening a real socket, keeping the test hermetic.
    """
    predict_app = FastAPI()

    @predict_app.post(PREDICT_PATH)
    async def fake_predict(payload: dict) -> dict:
        return {
            "taskType": "asr",
            "output": [{"source": "হ্যালো"}],
            "time_taken": 0.1,
        }

    real_async_client = httpx.AsyncClient

    def fake_async_client(*args, **kwargs):
        return real_async_client(
            transport=httpx.ASGITransport(app=predict_app),
            base_url="http://internal",
        )

    monkeypatch.setattr(
        "src.api.v1.live_cc.router.httpx.AsyncClient", fake_async_client
    )

    app = FastAPI()
    app.include_router(live_cc_router)
    return TestClient(app)


def test_live_cc_ws_emits_caption_per_chunk(live_cc_client):
    chunk_samples = int(
        settings.LIVE_CC_INPUT_SAMPLE_RATE * settings.LIVE_CC_CHUNK_SECONDS
    )
    pcm = np.zeros(chunk_samples, dtype=np.int16).tobytes()

    with live_cc_client.websocket_connect("/api/v1/live-cc/ws") as ws:
        ws.send_bytes(pcm)
        message = ws.receive_json()

    assert message == {"text": "হ্যালো", "is_final": True}


def test_live_cc_ws_buffers_partial_chunks(live_cc_client):
    """Half a chunk (1.5s) already crosses one interim step (1.0s), so the
    first half triggers an interim caption before the second half finalizes it.
    """
    chunk_samples = int(
        settings.LIVE_CC_INPUT_SAMPLE_RATE * settings.LIVE_CC_CHUNK_SECONDS
    )
    half_pcm = np.zeros(chunk_samples // 2, dtype=np.int16).tobytes()

    with live_cc_client.websocket_connect("/api/v1/live-cc/ws") as ws:
        ws.send_bytes(half_pcm)
        interim = ws.receive_json()
        ws.send_bytes(half_pcm)
        final = ws.receive_json()

    assert interim == {"text": "হ্যালো", "is_final": False}
    assert final == {"text": "হ্যালো", "is_final": True}


def test_live_cc_ws_emits_interim_captions_before_final(live_cc_client):
    """Steps through interim intervals, then sends the remainder to cross the
    final chunk boundary.
    """
    interim_samples = int(
        settings.LIVE_CC_INPUT_SAMPLE_RATE * settings.LIVE_CC_INTERIM_INTERVAL_SECONDS
    )
    step_pcm = np.zeros(interim_samples, dtype=np.int16).tobytes()

    chunk_seconds = settings.LIVE_CC_CHUNK_SECONDS
    interim_seconds = settings.LIVE_CC_INTERIM_INTERVAL_SECONDS
    num_interim_steps = int(chunk_seconds / interim_seconds) - 1

    with live_cc_client.websocket_connect("/api/v1/live-cc/ws") as ws:
        messages = []
        for _ in range(num_interim_steps):
            ws.send_bytes(step_pcm)
            messages.append(ws.receive_json())

        remaining_samples = (
            int(settings.LIVE_CC_INPUT_SAMPLE_RATE * chunk_seconds)
            - interim_samples * num_interim_steps
        )
        ws.send_bytes(np.zeros(remaining_samples, dtype=np.int16).tobytes())
        messages.append(ws.receive_json())

    assert all(m["is_final"] is False for m in messages[:-1])
    assert messages[-1]["is_final"] is True


@pytest.fixture
def speaker_gate_client(monkeypatch):
    """Speaker-gate test double: /internal/speaker/embed derives its fake
    embedding from the actual PCM sample value in the request (positive vs
    negative constant signal -> two distinct, deterministic embeddings). It
    deliberately does NOT depend on any shared mutable state the test sets
    from the main thread — TestClient's websocket runs the ASGI app on a
    background task, so there's no guaranteed ordering between the test
    thread mutating external state and the server actually consuming a given
    chunk; deriving the embedding from the request body itself sidesteps that
    race entirely. /predict logs each call with an incrementing counter
    rather than a fixed transcript, so a gated chunk that wrongly reaches
    transcription is provable by the *number* in the response, not by
    asserting the absence of a message (which would risk a hanging blocking
    receive if nothing arrives).
    """
    monkeypatch.setattr(settings, "SPEAKER_GATE_ENABLED", True)
    monkeypatch.setattr(settings, "SPEAKER_ENROLL_SECONDS", 1.0)
    monkeypatch.setattr(settings, "LIVE_CC_INTERIM_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(settings, "SPEAKER_SIMILARITY_THRESHOLD", 0.5)

    call_log = []
    predict_app = FastAPI()

    @predict_app.post(PREDICT_PATH)
    async def fake_predict(payload: dict) -> dict:
        call_log.append(1)
        return {
            "taskType": "asr",
            "output": [{"source": f"chunk-{len(call_log)}"}],
            "time_taken": 0.1,
        }

    @predict_app.post("/internal/speaker/embed")
    async def fake_embed(payload: dict) -> dict:
        wav_bytes = base64.b64decode(payload["audio_content"])
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            first_frame = wf.readframes(1)
        sample = int.from_bytes(first_frame, byteorder="little", signed=True)
        return {"embedding": [1.0, 0.0] if sample >= 0 else [0.0, 1.0]}

    real_async_client = httpx.AsyncClient

    def fake_async_client(*args, **kwargs):
        return real_async_client(
            transport=httpx.ASGITransport(app=predict_app),
            base_url="http://internal",
        )

    monkeypatch.setattr(
        "src.api.v1.live_cc.router.httpx.AsyncClient", fake_async_client
    )

    app = FastAPI()
    app.include_router(live_cc_router)
    return TestClient(app)


def test_live_cc_ws_speaker_gate_drops_non_matching_chunk(speaker_gate_client):
    """Enrolls on the first second of audio (positive-sample signal), then
    verifies a 3s chunk built from a different (negative-sample) signal never
    reaches transcription — proven by the next matching chunk being logged as
    call #1, not #2.
    """
    client = speaker_gate_client
    input_sr = settings.LIVE_CC_INPUT_SAMPLE_RATE
    matching_pcm = np.full(input_sr, 100, dtype=np.int16).tobytes()
    different_pcm = np.full(input_sr, -100, dtype=np.int16).tobytes()

    with client.websocket_connect("/api/v1/live-cc/ws") as ws:
        ws.send_bytes(matching_pcm)  # enrolls on the positive-sample signal

        ws.send_bytes(different_pcm)
        ws.send_bytes(different_pcm)  # completes a 3s chunk that should be dropped

        ws.send_bytes(matching_pcm)
        ws.send_bytes(matching_pcm)
        ws.send_bytes(matching_pcm)  # completes the next 3s chunk, should transcribe

        message = ws.receive_json()

    assert message == {"text": "chunk-1", "is_final": True}
