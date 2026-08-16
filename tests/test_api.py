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
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from main import create_gateway_app
from src.api.client import PREDICT_PATH
from src.api.v1.asr.router import router as asr_router
from src.api.v1.asr.schema import AsrRequest
from src.api.v1.live_cc.router import router as live_cc_router
from src.core.config import settings
from src.litserver.engine.conformer import ConformerEngine
from src.litserver.engine.whisper import WhisperEngine
from src.litserver.litapi import ASRLitAPI
from src.utils.audio import decode_base64_audio
from src.utils.bn_text_repair import repair_stray_vowel_signs
from src.utils.itn import bengali_numerals_to_digits as itn


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
    assert body["model"] == settings.ACTIVE_MODEL_NAME


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


def test_is_non_speech_flags_sparse_hallucinations():
    """Outputs actually observed from silence, noise, tone and music."""
    assert ConformerEngine.is_non_speech("তেন", 5.0)
    assert ConformerEngine.is_non_speech("ত", 3.0)
    assert ConformerEngine.is_non_speech("সগগগগগগ্গগ্গেন", 5.0)
    assert ConformerEngine.is_non_speech("স", 15.0)


def test_is_non_speech_keeps_real_speech():
    """Real FLEURS speech never fell below 4.46 chars/sec over 1322 clips."""
    sentence = "জার্মানির অনেক বেক করা খাবারগুলিতে বাদাম পাওয়া যায়"
    assert not ConformerEngine.is_non_speech(sentence, 8.0)
    # A short utterance in a short segment is dense enough to keep.
    assert not ConformerEngine.is_non_speech("হ্যালো", 1.0)


def test_is_non_speech_absolute_cap_protects_short_real_phrases():
    """A brief real phrase in a long quiet segment is sparse, but exceeding the
    character cap keeps it — only tiny *and* sparse output is discarded.
    """
    phrase = "আমি তোমাকে বলেছি ভাই"  # > NON_SPEECH_MAX_CHARS
    assert len("".join(phrase.split())) > 15
    assert not ConformerEngine.is_non_speech(phrase, 18.0)


def test_is_non_speech_ignores_empty_and_bad_duration():
    assert not ConformerEngine.is_non_speech("", 5.0)
    assert not ConformerEngine.is_non_speech("   ", 5.0)
    assert not ConformerEngine.is_non_speech("তেন", 0.0)


def test_transcribe_drops_non_speech_segment_but_keeps_speech():
    """The reported symptom: a music/silence tail became its own segment and
    appended a stray character to an otherwise-correct transcript.
    """
    sr = 16000
    engine = ConformerEngine(model_name="dummy", device="cpu", max_segment_seconds=18.0)
    engine.model = MagicMock()
    engine.model.transcribe.return_value = [
        "আমি ভাত খেয়ে বাড়িতে চলে গিয়েছিলাম আজকে সকালে",
        "ত",
    ]

    audio = np.zeros(int(sr * 30), dtype=np.float32)
    result = engine.transcribe([audio], batch_size=2, sample_rate=sr)

    assert result == ["আমি ভাত খেয়ে বাড়িতে চলে গিয়েছিলাম আজকে সকালে"]
    assert not result[0].endswith("ত ")


def test_whisper_engine_drops_non_speech_hallucination():
    """Regression: this repo's Whisper checkpoint decodes pure silence to the
    literal string "<>", which then popped up repeatedly in live captions
    since WhisperEngine (unlike ConformerEngine) never filtered it.
    """
    sr = 16000
    engine = WhisperEngine(model_name="dummy", device="cpu", max_segment_seconds=28.0)
    engine.processor = MagicMock()
    engine.processor.return_value.input_features = torch.zeros(1, 1, 1)
    engine.processor.batch_decode.return_value = ["<>"]
    engine.model = MagicMock()
    engine.model.generate.return_value = torch.zeros(1, 1, dtype=torch.long)
    engine.decoder_input_ids = torch.zeros(1, 1, dtype=torch.long)

    audio = np.zeros(int(sr * 3), dtype=np.float32)
    result = engine.transcribe([audio], batch_size=1, sample_rate=sr)

    assert result == [""]


def test_transcribe_non_speech_drop_can_be_disabled():
    sr = 16000
    engine = ConformerEngine(
        model_name="dummy",
        device="cpu",
        max_segment_seconds=18.0,
        drop_non_speech=False,
    )
    engine.model = MagicMock()
    engine.model.transcribe.return_value = ["আজকে আমি বাড়িতে গিয়েছিলাম", "ত"]

    audio = np.zeros(int(sr * 30), dtype=np.float32)
    result = engine.transcribe([audio], batch_size=2, sample_rate=sr)

    assert result[0].endswith("ত")


def test_itn_converts_compound_hundreds_and_decimals():
    assert itn("আটশো দুই দশমিক এগারো এন") == "802.11 এন"
    assert itn("দুই দশমিক চার গিগাহার্জ") == "2.4 গিগাহার্জ"


def test_itn_converts_thousands_and_separated_hundreds():
    assert itn("এক হাজার নয় শো ঊননব্বই সালে") == "1989 সালে"
    assert itn("এক শো আশি ডিগ্রি") == "180 ডিগ্রি"
    assert itn("দুই লাখ") == "200000"
    assert itn("তিন কোটি") == "30000000"


def test_itn_leaves_small_bare_numerals_spelled_out():
    """Small numbers in running prose are normally written as words, so
    converting them costs more than it gains (measured on FLEURS).
    """
    assert itn("এক ব্যক্তি হাঁটছিলেন") == "এক ব্যক্তি হাঁটছিলেন"
    assert itn("দুই") == "দুই"
    # ...but a scale word means it reads as a real quantity.
    assert itn("দুই হাজার") == "2000"


def test_itn_leaves_non_numeric_text_byte_identical():
    for text in ("কোন সংখ্যা নেই এখানে", "", "   ", "ইরানে বড় ধরনের হামলা"):
        assert itn(text) == text


def test_itn_unknown_tokens_pass_through_unchanged():
    """The allowlist design means an unrecognised word must end the run and
    survive verbatim — a missed conversion, never corrupted text.
    """
    assert itn("ফুটবল খেলা") == "ফুটবল খেলা"
    assert itn("দশ ফুটবল বিশ") == "10 ফুটবল 20"


def test_itn_preserves_surrounding_whitespace():
    assert itn("  দশ  টাকা  ") == "  10  টাকা  "


def test_itn_min_value_gate_is_tunable():
    assert itn("দুই", min_value=0) == "2"
    assert itn("দুই", min_value=10) == "দুই"


def test_lit_api_applies_itn_to_response(lit_api, monkeypatch):
    monkeypatch.setattr(settings, "ITN_ENABLED", True)
    lit_api.engine.transcribe.return_value = ["এক শো আশি ডিগ্রি"]
    request = AsrRequest(audio=[{"audioContent": make_wav_base64()}])
    prediction = lit_api.predict(lit_api.decode_request(request))
    assert lit_api.encode_response(prediction).output[0].source == "180 ডিগ্রি"


def test_lit_api_itn_can_be_disabled(lit_api, monkeypatch):
    monkeypatch.setattr(settings, "ITN_ENABLED", False)
    lit_api.engine.transcribe.return_value = ["এক শো আশি ডিগ্রি"]
    request = AsrRequest(audio=[{"audioContent": make_wav_base64()}])
    prediction = lit_api.predict(lit_api.decode_request(request))
    assert (
        lit_api.encode_response(prediction).output[0].source == "এক শো আশি ডিগ্রি"
    )


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


def test_repair_stray_vowel_signs_reattaches_floating_matra():
    """Regression: wav2vec2 CTC decoding sometimes emits a dependent vowel
    sign as its own space-separated "word" instead of attached to the
    preceding consonant.
    """
    assert repair_stray_vowel_signs("কাছ েেে কিছুই") == "কাছে কিছুই"
    assert repair_stray_vowel_signs("স্পা ে প্রাচীর") == "স্পাে প্রাচীর"


def test_repair_stray_vowel_signs_leaves_normal_text_unchanged():
    for text in ("আজকে আমি বাড়িতে গিয়েছিলাম", "", "   "):
        assert repair_stray_vowel_signs(text) == text


def test_lit_api_applies_vowel_repair_only_for_wav2vec2(lit_api, monkeypatch):
    monkeypatch.setattr(settings, "ENGINE", "wav2vec2")
    monkeypatch.setattr(settings, "ITN_ENABLED", False)
    lit_api.engine.transcribe.return_value = ["কাছ েেে কিছুই"]
    request = AsrRequest(audio=[{"audioContent": make_wav_base64()}])
    prediction = lit_api.predict(lit_api.decode_request(request))
    assert lit_api.encode_response(prediction).output[0].source == "কাছে কিছুই"

    monkeypatch.setattr(settings, "ENGINE", "conformer")
    prediction = lit_api.predict(lit_api.decode_request(request))
    assert lit_api.encode_response(prediction).output[0].source == "কাছ েেে কিছুই"


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
    # drop_non_speech off: the one-character stub transcripts below are far too
    # sparse to pass the non-speech gate, and this test is about splitting.
    engine = ConformerEngine(
        model_name="dummy",
        device="cpu",
        max_segment_seconds=1.0,
        drop_non_speech=False,
    )
    engine.model = MagicMock()
    engine.model.transcribe.return_value = ["a", "b", "c"]

    sr = 16000
    audio = np.zeros(int(sr * 2.5), dtype=np.float32)
    result = engine.transcribe([audio], batch_size=4, sample_rate=sr)

    assert result == ["a b c"]
    assert len(engine.model.transcribe.call_args.kwargs["audio"]) == 3


def test_asr_engine_leaves_short_audio_unsplit():
    # See above: stub transcript is too short for the non-speech gate.
    engine = ConformerEngine(
        model_name="dummy",
        device="cpu",
        max_segment_seconds=18.0,
        drop_non_speech=False,
    )
    engine.model = MagicMock()
    engine.model.transcribe.return_value = ["hello"]

    sr = 16000
    audio = np.zeros(int(sr * 2.0), dtype=np.float32)
    result = engine.transcribe([audio], batch_size=4, sample_rate=sr)

    assert result == ["hello"]
    assert len(engine.model.transcribe.call_args.kwargs["audio"]) == 1


def test_asr_engine_respects_custom_sample_rate_for_segment_length():
    """max_segment_seconds is interpreted against the request's sample rate,
    not a hardcoded 16kHz. Exact boundaries depend on where split() finds a
    quiet point, so this asserts the length ceiling rather than a fixed count.
    """
    engine = ConformerEngine(model_name="dummy", device="cpu", max_segment_seconds=1.0)
    engine.model = MagicMock()
    engine.model.transcribe.side_effect = lambda audio, **kw: [
        f"s{i}" for i in range(len(audio))
    ]

    audio = np.zeros(8000 * 2, dtype=np.float32)
    engine.transcribe([audio], batch_size=4, sample_rate=8000)

    segments = engine.model.transcribe.call_args.kwargs["audio"]
    assert len(segments) > 1
    assert all(len(seg) <= 8000 for seg in segments)
    assert sum(len(seg) for seg in segments) == audio.size


def test_asr_engine_split_prefers_quiet_cut_points():
    """A hard cut landing mid-word truncates it in both neighbouring segments
    and CTC tends to drop it entirely. Boundaries must land in the pauses.
    """
    sr = 16000
    # 1s of tone, then 0.5s of silence, repeating.
    block = np.concatenate(
        [
            np.sin(2 * np.pi * 220 * np.arange(sr) / sr).astype(np.float32),
            np.zeros(sr // 2, dtype=np.float32),
        ]
    )
    audio = np.tile(block, 40)

    segments = ConformerEngine.split(audio, sr * 18, sr * 2)

    period = sr + sr // 2
    boundaries = np.cumsum([len(s) for s in segments])[:-1]
    assert len(boundaries) > 0
    # phase >= sr means the boundary sits in a silent gap, not in the tone.
    assert all(b % period >= sr for b in boundaries)


def test_asr_engine_split_is_lossless_and_bounded():
    sr = 16000
    audio = np.random.RandomState(0).randn(sr * 87).astype(np.float32)

    segments = ConformerEngine.split(audio, sr * 18, sr * 2)

    assert np.array_equal(np.concatenate(segments), audio)
    assert all(len(s) <= sr * 18 for s in segments)


def test_asr_engine_split_without_search_uses_fixed_cuts():
    """boundary_search_samples=0 must reproduce plain fixed-size slicing."""
    sr = 16000
    audio = np.random.RandomState(0).randn(sr * 50).astype(np.float32)
    max_samples = sr * 18

    segments = ConformerEngine.split(audio, max_samples, 0)
    expected = [
        audio[i : i + max_samples] for i in range(0, audio.size, max_samples)
    ]

    assert len(segments) == len(expected)
    assert all(np.array_equal(a, b) for a, b in zip(segments, expected))


def test_asr_engine_split_keeps_segments_long_on_flat_audio():
    """Flat-energy audio ties on every frame; the cut must stay at the far end
    of the search window rather than always jumping to its start.
    """
    sr = 16000
    audio = np.zeros(sr * 50, dtype=np.float32)

    segments = ConformerEngine.split(audio, sr * 18, sr * 2)

    assert len(segments[0]) > sr * 17


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

