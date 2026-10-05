import base64
import io
import json
import pathlib
import shutil
import subprocess
import tempfile
import wave
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import numpy as np
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from loguru import logger
from pydantic import ValidationError

from main import create_gateway_app
from src.api.client import PREDICT_PATH, SYNTHESIZE_PATH
from src.api.v1.asr.schema import AsrRequest
from src.api.v1.tts.cache import SpeechCache
from src.api.v1.tts.router import router as tts_router
from src.api.v1.tts.router import speech_cache
from src.api.v1.tts.schema import TtsRequest
from src.core.config import settings
from src.litserver.base import Audio, BaseTTSEngine
from src.litserver.litapi import TTS_API_PATH, ASRLitAPI, TTSLitAPI
from src.litserver.parler.chunking import IncrementalTextChunker, chunk_text
from src.litserver.parler.voices import VOICES as TTS_VOICES
from src.litserver.zipformer.engine import ZipformerEngine
from src.litserver.zipformer.layouts import DEFAULT, K2_FSA, VOSK_BN
from src.utils.audio import decode_base64_audio
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
def asr_client(monkeypatch, tmp_path):
    """A fake /predict on a separate app, reached in-process by monkeypatching
    the client's httpx.AsyncClient so no real socket opens.
    """
    predict_app = FastAPI()
    captured: dict = {}

    @predict_app.post(PREDICT_PATH)
    async def fake_predict(payload: dict) -> dict:
        captured["payload"] = payload
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
    monkeypatch.setattr(settings, "TRACE_DIR", str(tmp_path / "traces"))

    app = create_gateway_app()
    app.state.captured = captured
    return TestClient(app)


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
    """Regression test: webm/opus needs the ffmpeg fallback's real file path,
    which an in-memory BytesIO can't provide.
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
    assert ZipformerEngine.is_non_speech("তেন", 5.0)
    assert ZipformerEngine.is_non_speech("ত", 3.0)
    assert ZipformerEngine.is_non_speech("সগগগগগগ্গগ্গেন", 5.0)
    assert ZipformerEngine.is_non_speech("স", 15.0)


def test_is_non_speech_keeps_real_speech():
    """Real FLEURS speech never fell below 4.46 chars/sec over 1322 clips."""
    sentence = "জার্মানির অনেক বেক করা খাবারগুলিতে বাদাম পাওয়া যায়"
    assert not ZipformerEngine.is_non_speech(sentence, 8.0)
    assert not ZipformerEngine.is_non_speech("হ্যালো", 1.0)


def test_is_non_speech_absolute_cap_protects_short_real_phrases():
    """Only output that is both tiny and sparse is discarded."""
    phrase = "আমি তোমাকে বলেছি ভাই"
    assert len("".join(phrase.split())) > 15
    assert not ZipformerEngine.is_non_speech(phrase, 18.0)


def test_is_non_speech_ignores_empty_and_bad_duration():
    assert not ZipformerEngine.is_non_speech("", 5.0)
    assert not ZipformerEngine.is_non_speech("   ", 5.0)
    assert not ZipformerEngine.is_non_speech("তেন", 0.0)


def test_is_non_speech_flags_literal_hallucination_on_short_segments():
    """"<>" is too dense on a short segment to trip the sparse-output
    heuristic, so it must be caught as a known literal.
    """
    assert ZipformerEngine.is_non_speech("<>", 0.5)
    assert ZipformerEngine.is_non_speech("<>", 5.0)


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
    request = AsrRequest(
        config={"language": {"sourceLanguage": "bn"}},
        audio=[{"audioContent": make_wav_base64()}],
    )
    prediction = lit_api.predict(lit_api.decode_request(request))
    assert lit_api.encode_response(prediction).output[0].source == "180 ডিগ্রি"


def test_lit_api_itn_can_be_disabled(lit_api, monkeypatch):
    monkeypatch.setattr(settings, "ITN_ENABLED", False)
    lit_api.engine.transcribe.return_value = ["এক শো আশি ডিগ্রি"]
    request = AsrRequest(
        config={"language": {"sourceLanguage": "bn"}},
        audio=[{"audioContent": make_wav_base64()}],
    )
    prediction = lit_api.predict(lit_api.decode_request(request))
    assert (
        lit_api.encode_response(prediction).output[0].source == "এক শো আশি ডিগ্রি"
    )


def test_asr_request_requires_the_nested_config():
    """config.language is required, not defaulted: the contract this mirrors
    always sends it, and silently substituting a default would hide a caller
    that got the shape wrong."""
    with pytest.raises(ValidationError):
        AsrRequest(audio=[{"audioContent": "abc"}])
    with pytest.raises(ValidationError):
        AsrRequest(config={}, audio=[{"audioContent": "abc"}])

def test_asr_request_carries_the_nested_language():
    req = AsrRequest(
        config={"language": {"sourceLanguage": "bn"}},
        audio=[{"audioContent": "abc"}],
    )
    assert req.config.language.sourceLanguage == "bn"


@pytest.fixture
def lit_api():
    api = ASRLitAPI(max_batch_size=1, api_path=PREDICT_PATH)
    api.engine = MagicMock()
    api.engine.model = object()
    api.engine.transcribe.return_value = ["হ্যালো"]
    return api


def test_lit_api_full_cycle(lit_api):
    request = AsrRequest(
        config={"language": {"sourceLanguage": "bn"}},
        audio=[{"audioContent": make_wav_base64()}],
    )
    decoded = lit_api.decode_request(request)
    assert len(decoded["audios"]) == 1

    prediction = lit_api.predict(decoded)
    lit_api.engine.transcribe.assert_called_once()

    response = lit_api.encode_response(prediction)
    assert response.taskType == "asr"
    assert response.output[0].source == "হ্যালো"
    assert response.time_taken >= 0


class FakeTTSEngine(BaseTTSEngine):
    """Records what it was asked to say and returns one flat second of audio
    per call. Subclasses the real contract so it inherits speak()/join()
    rather than reimplementing them."""

    voices = TTS_VOICES

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.calls = []

    def load(self) -> None:
        pass

    def synthesize(self, text, voice="", description=None):
        voice = voice or "Aditi"
        if voice not in self.voices:
            raise ValueError(f"unknown voice: {voice}")
        if not text.strip():
            raise ValueError("text must not be empty")
        self.calls.append((text, voice, description))
        return Audio(np.full(self.sample_rate, 0.5, dtype=np.float32), self.sample_rate)


class ChunkingFakeTTSEngine(FakeTTSEngine):
    """A fake that splits like Parler does, for the chunk-and-join path."""

    max_chars = 160

    def speak_stream(self, text, voice="", description=None):
        for chunk in chunk_text(text, max_chars=self.max_chars):
            yield self.synthesize(chunk, voice, description)


@pytest.fixture
def tts_lit_api():
    api = TTSLitAPI(max_batch_size=1, api_path=TTS_API_PATH)
    api.engine = FakeTTSEngine()
    return api


def test_tts_lit_api_full_cycle(tts_lit_api):
    """One clause in, one NDJSON line of raw PCM out."""
    request = TtsRequest(input="আমি ভালো আছি।")
    lines = list(
        tts_lit_api.encode_response(
            tts_lit_api.predict(tts_lit_api.decode_request(request))
        )
    )
    assert len(lines) == 1
    assert lines[0].endswith("\n")

    chunk = json.loads(lines[0])
    assert chunk["index"] == 0
    assert chunk["sampleRate"] == 44100
    assert chunk["voice"] == settings.TTS_VOICE

    pcm = base64.b64decode(chunk["audioContent"])
    assert len(pcm) == 44100 * 2


def test_tts_lit_api_streams_each_clause_as_its_own_line(tts_lit_api):
    """The point of the streaming path: a three-clause reply reaches the
    caller as three lines, the first available before the last is generated,
    rather than as one buffer at the end."""
    tts_lit_api.engine = ChunkingFakeTTSEngine()
    text = "এক দুই তিন। চার পাঁচ ছয়। সাত আট নয়।"
    lines = list(
        tts_lit_api.encode_response(
            tts_lit_api.predict(tts_lit_api.decode_request(TtsRequest(input=text)))
        )
    )
    assert [json.loads(line)["index"] for line in lines] == [0, 1, 2]

    gap = round(0.08 * 44100)
    sizes = [len(base64.b64decode(json.loads(line)["audioContent"])) for line in lines]
    assert sizes == [44100 * 2, (44100 + gap) * 2, (44100 + gap) * 2]


def test_tts_stream_concatenates_into_exactly_what_speak_returns():
    """What lets one streaming worker serve the whole-buffer endpoint too:
    joining the stream must reproduce speak() sample for sample, gaps and
    all."""
    engine = ChunkingFakeTTSEngine()
    text = "এক দুই তিন। চার পাঁচ ছয়। সাত আট নয়।"
    streamed = np.concatenate(
        [part.samples for part in engine.stream_playable(text)]
    )
    whole = engine.speak(text).samples
    assert streamed.size == whole.size
    assert np.array_equal(streamed, whole)


def test_tts_lit_api_hands_the_whole_request_to_the_engine(tts_lit_api):
    """Splitting is the engine's business now, so a plain engine sees the text
    whole and the LitAPI stays model-agnostic."""
    text = "এক দুই তিন। চার পাঁচ ছয়। সাত আট নয়।"
    list(tts_lit_api.predict(tts_lit_api.decode_request(TtsRequest(input=text))))
    assert [call[0] for call in tts_lit_api.engine.calls] == [text]


def test_chunking_engine_splits_and_joins_with_gaps():
    """The Parler-shaped path: speak() splits on clause boundaries, synthesizes
    each and joins with a gap between parts but not around them."""
    api = TTSLitAPI(max_batch_size=1, api_path=TTS_API_PATH)
    api.engine = ChunkingFakeTTSEngine()
    text = "এক দুই তিন। চার পাঁচ ছয়। সাত আট নয়।"
    parts = list(api.predict(api.decode_request(TtsRequest(input=text))))

    assert len(api.engine.calls) == 3
    expected = 3 * 44100 + 2 * round(0.08 * 44100)
    assert sum(part["audio"].samples.size for part in parts) == expected


def test_tts_lit_api_passes_voice_and_description_through(tts_lit_api):
    request = TtsRequest(input="পরীক্ষা।", voice="Arjun", description="  slow and calm  ")
    list(tts_lit_api.predict(tts_lit_api.decode_request(request)))
    assert tts_lit_api.engine.calls == [("পরীক্ষা।", "Arjun", "  slow and calm  ")]


def test_tts_lit_api_unknown_voice_is_422_not_500(tts_lit_api):
    decoded = tts_lit_api.decode_request(TtsRequest(input="পরীক্ষা।"))
    decoded["voice"] = "Nobody"
    with pytest.raises(HTTPException) as excinfo:
        list(tts_lit_api.predict(decoded))
    assert excinfo.value.status_code == 422


def test_tts_predict_is_a_generator_function(tts_lit_api):
    """LitServe refuses to start a stream=True LitAPI unless predict itself
    contains yield, so this shape is load-bearing, not a style choice."""
    import inspect

    assert inspect.isgeneratorfunction(type(tts_lit_api).predict)
    assert inspect.isgeneratorfunction(type(tts_lit_api).encode_response)


def test_tts_predict_rejects_the_voice_before_emitting_any_chunk(tts_lit_api):
    """The check must land before the first yield: once a chunk is out the
    response already carries a 200 and the status cannot be changed."""
    decoded = tts_lit_api.decode_request(TtsRequest(input="পরীক্ষা।"))
    decoded["voice"] = "Nobody"
    stream = tts_lit_api.predict(decoded)
    with pytest.raises(HTTPException):
        next(stream)
    assert tts_lit_api.engine.calls == []


def test_tts_request_rejects_blank_input():
    with pytest.raises(ValidationError):
        TtsRequest(input="   ")


def test_tts_engine_join_rejects_mixed_sample_rates():
    parts = [
        Audio(np.zeros(4, dtype=np.float32), 44100),
        Audio(np.zeros(4, dtype=np.float32), 16000),
    ]
    with pytest.raises(ValueError):
        BaseTTSEngine.join(parts)


def test_audio_pcm_clips_instead_of_wrapping():
    """Without the clip, a sample above 1.0 overflows int16 into a loud
    negative: an audible pop rather than clean saturation."""
    pcm = Audio(np.array([2.0, -2.0], dtype=np.float32), 16000).pcm_s16le()
    assert np.frombuffer(pcm, dtype="<i2").tolist() == [32767, -32767]


def test_chunker_hard_splits_text_with_no_punctuation():
    long_text = " ".join(["শব্দ"] * 200)
    chunks = chunk_text(long_text, max_chars=80)
    assert len(chunks) > 1
    assert all(len(chunk) <= 80 for chunk in chunks)
    assert "".join(chunks.copy()).replace(" ", "") == long_text.replace(" ", "")


def test_chunker_holds_incomplete_clause_until_flush():
    chunker = IncrementalTextChunker(max_chars=160)
    assert chunker.feed("আমি ভালো") == []
    assert chunker.feed(" আছি।") == ["আমি ভালো আছি।"]
    assert chunker.feed("বাকি অংশ") == []
    assert chunker.flush() == ["বাকি অংশ"]


FAKE_TTS_CHUNKS = 2
FAKE_TTS_CHUNK_FRAMES = 1000


@pytest.fixture
def tts_client(monkeypatch):
    """A fake LitServe /synthesize, wired in through the shared client.

    Pins TTS_ENABLED rather than trusting the default, so these tests keep
    testing the enabled path whichever way that default is set."""
    monkeypatch.setattr(settings, "TTS_ENABLED", True)
    speech_cache.clear()
    fake = FastAPI()
    calls: list[dict] = []

    @fake.post(SYNTHESIZE_PATH)
    async def synthesize(payload: dict) -> StreamingResponse:
        """Two NDJSON chunks of raw PCM, the shape TTSLitAPI now streams."""
        calls.append(payload)

        async def lines():
            for index in range(FAKE_TTS_CHUNKS):
                silence = np.zeros(FAKE_TTS_CHUNK_FRAMES, dtype="<i2")
                yield json.dumps(
                    {
                        "index": index,
                        "audioContent": base64.b64encode(
                            silence.tobytes()
                        ).decode("utf-8"),
                        "sampleRate": 44100,
                        "voice": payload.get("voice") or "Aditi",
                        "time_taken": 0.1 * (index + 1),
                    }
                ) + "\n"

        return StreamingResponse(lines(), media_type="application/x-ndjson")

    transport = httpx.ASGITransport(app=fake)
    monkeypatch.setattr(
        "src.api.client.get_litserve_client",
        lambda: httpx.AsyncClient(transport=transport, base_url="http://litserver:8000"),
    )
    app = FastAPI()
    app.include_router(tts_router)
    client = TestClient(app)
    client.synthesize_calls = calls
    """Every payload the model server was asked to synthesize, so a test can
    assert that a second identical request never reached it."""
    return client


def test_tts_synthesize_audio_endpoint_returns_playable_wav(tts_client):
    resp = tts_client.post("/v1/audio/speech", json={"input": "হ্যালো"})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    with wave.open(io.BytesIO(resp.content), "rb") as wf:
        assert wf.getframerate() == 44100


def test_tts_synthesize_rejects_unknown_voice_before_hitting_the_model(tts_client):
    resp = tts_client.post(
        "/v1/audio/speech", json={"input": "হ্যালো", "voice": "Nobody"}
    )
    assert resp.status_code == 422


def test_tts_routes_return_503_when_disabled(tts_client, monkeypatch):
    monkeypatch.setattr(settings, "TTS_ENABLED", False)
    resp = tts_client.post("/v1/audio/speech", json={"input": "হ্যালো"})
    assert resp.status_code == 503
    assert tts_client.post("/v1/audio/speech", json={"input": "x"}).status_code == 503


def _capture_zipformer_load(monkeypatch):
    """Record what ZipformerEngine.load() asks the Hub for, without network or
    a real recognizer. Returns (requested, recognizer_kwargs), both filled in
    once load() runs."""
    requested = []
    recognizer_kwargs = {}

    def fake_download(repo, filename):
        requested.append((repo, filename))
        return f"/fake/{filename}"

    def fake_from_transducer(**kwargs):
        recognizer_kwargs.update(kwargs)
        return MagicMock()

    monkeypatch.setattr(
        "src.litserver.zipformer.engine.hf_hub_download", fake_download
    )
    monkeypatch.setattr(
        "sherpa_onnx.OnlineRecognizer.from_transducer",
        staticmethod(fake_from_transducer),
    )
    return requested, recognizer_kwargs


def test_zipformer_engine_defaults_to_the_vosk_repo_layout(monkeypatch):
    requested, recognizer_kwargs = _capture_zipformer_load(monkeypatch)
    ZipformerEngine(model_name="alphacep/vosk-model-small-streaming-bn").load(
        warmup_seconds=0
    )
    assert requested == [
        ("alphacep/vosk-model-small-streaming-bn", "am-onnx/encoder.onnx"),
        ("alphacep/vosk-model-small-streaming-bn", "am-onnx/decoder.onnx"),
        ("alphacep/vosk-model-small-streaming-bn", "am-onnx/joiner.onnx"),
        ("alphacep/vosk-model-small-streaming-bn", "lang/tokens.txt"),
    ]
    assert recognizer_kwargs["tokens"] == "/fake/lang/tokens.txt"
    assert DEFAULT is VOSK_BN


def test_zipformer_engine_downloads_the_layout_it_was_given(monkeypatch):
    """A checkpoint with a different repo layout is served by passing another
    descriptor from src/litserver/zipformer/layouts.py, with no engine change."""
    requested, recognizer_kwargs = _capture_zipformer_load(monkeypatch)
    ZipformerEngine(
        model_name="k2-fsa/sherpa-onnx-streaming-zipformer-bilingual-zh-en",
        layout=K2_FSA,
    ).load(warmup_seconds=0)

    assert [path for _, path in requested] == [
        "encoder-epoch-99-avg-1.onnx",
        "decoder-epoch-99-avg-1.onnx",
        "joiner-epoch-99-avg-1.onnx",
        "tokens.txt",
    ]
    assert recognizer_kwargs["encoder"] == "/fake/encoder-epoch-99-avg-1.onnx"
    assert recognizer_kwargs["tokens"] == "/fake/tokens.txt"


def test_zipformer_layouts_are_immutable():
    """Layouts are shared module-level defaults; a mutable one would let a
    single engine instance rewrite what every later instance downloads."""
    with pytest.raises(Exception):
        VOSK_BN.encoder = "somewhere/else.onnx"


class FakeRecognizer:
    """Stands in for sherpa_onnx.OnlineRecognizer, recording frames fed to the
    one stream it hands out so session lifetime is observable."""

    def __init__(self, transcripts):
        self.transcripts = list(transcripts)
        self.frames = []
        self.resets = 0
        self.endpoint = False
        self.finished = False

    def create_stream(self):
        recognizer = self

        class Stream:
            def accept_waveform(self, sample_rate, samples):
                recognizer.frames.append(len(samples))

            def input_finished(self):
                recognizer.finished = True

        return Stream()

    def is_ready(self, stream):
        return False

    def decode_stream(self, stream):
        pass

    def get_result(self, stream):
        return self.transcripts[min(len(self.frames), len(self.transcripts)) - 1]

    def is_endpoint(self, stream):
        return self.endpoint

    def reset(self, stream):
        self.resets += 1
        self.transcripts = [""]


def _session(transcripts):
    engine = ZipformerEngine(model_name="dummy")
    engine.recognizer = FakeRecognizer(transcripts)
    return engine.stream(), engine.recognizer


def test_zipformer_stream_requires_load_first():
    with pytest.raises(RuntimeError, match="load\\(\\) must be called"):
        ZipformerEngine(model_name="dummy").stream()


def test_zipformer_session_decodes_incrementally_on_one_stream():
    """The point of a session: many frames, one stream — not one cold decode
    of the whole buffer per frame."""
    session, recognizer = _session(["আমি", "আমি ভালো", "আমি ভালো আছি"])
    frame = np.zeros(1600, dtype=np.float32)

    assert session.accept(frame) == "আমি"
    assert session.accept(frame) == "আমি ভালো"
    assert session.accept(frame) == "আমি ভালো আছি"
    assert recognizer.frames == [1600, 1600, 1600]


def test_zipformer_session_reports_endpoint_and_resets_for_the_next_turn():
    session, recognizer = _session(["আমি ভালো আছি"])
    session.accept(np.zeros(1600, dtype=np.float32))
    assert not session.is_endpoint()

    recognizer.endpoint = True
    assert session.is_endpoint()

    session.reset()
    assert recognizer.resets == 1
    assert session.text == ""


def test_zipformer_session_finish_flushes_the_tail():
    """Without the padding the decoder can still be holding the final word
    when a caller hangs up."""
    session, recognizer = _session(["শেষ"])
    session.accept(np.zeros(1600, dtype=np.float32))
    assert session.finish(tail_padding_seconds=0.5, sample_rate=16000) == "শেষ"
    assert recognizer.frames[-1] == 8000
    assert recognizer.finished


def test_speak_stream_yields_each_clause_before_the_whole_reply_is_done():
    """What makes streaming worth it: the first clause is available after one
    synthesize(), not after all of them."""
    engine = ChunkingFakeTTSEngine()
    stream = engine.speak_stream("এক দুই তিন। চার পাঁচ ছয়। সাত আট নয়।")

    first = next(stream)
    assert len(engine.calls) == 1
    assert first.samples.size == 44100

    rest = list(stream)
    assert len(engine.calls) == 3
    assert len(rest) == 2


def test_speak_joins_what_speak_stream_yields():
    """One override, both shapes: speak() is defined in terms of the stream."""
    engine = ChunkingFakeTTSEngine()
    audio = engine.speak("এক দুই তিন। চার পাঁচ ছয়।")
    assert len(engine.calls) == 2
    assert audio.samples.size == 2 * 44100 + round(0.08 * 44100)


def test_plain_engine_speak_stream_yields_once():
    engine = FakeTTSEngine()
    parts = list(engine.speak_stream("এক দুই তিন। চার পাঁচ ছয়।"))
    assert len(parts) == 1
    assert len(engine.calls) == 1


def test_zipformer_provider_defaults_to_cuda_and_is_configurable(monkeypatch):
    """The provider comes from ZIPFORMER_PROVIDER, not ACCELERATOR: it is a
    property of the installed wheel, not of the device LitServe assigns. Both
    directions are checked, since the two settings disagreeing is exactly the
    case that must not silently resolve to the accelerator."""
    from src.litserver.zipformer.engine import build as zipformer_build

    monkeypatch.setattr(settings, "ACCELERATOR", "cpu")
    assert zipformer_build("cpu").provider == "cuda"

    monkeypatch.setattr(settings, "ACCELERATOR", "cuda")
    monkeypatch.setattr(settings, "ZIPFORMER_PROVIDER", "cpu")
    assert zipformer_build("cuda:0").provider == "cpu"


def test_zipformer_passes_its_provider_to_the_recognizer(monkeypatch):
    _, recognizer_kwargs = _capture_zipformer_load(monkeypatch)
    ZipformerEngine(model_name="dummy", provider="cuda").load(warmup_seconds=0)
    assert recognizer_kwargs["provider"] == "cuda"


def test_zipformer_warns_when_cuda_is_asked_of_a_cpu_only_wheel(monkeypatch, caplog):
    """The failure mode this guards is silent: onnxruntime falls back to CPU
    rather than erroring, so a misconfigured deploy just runs slow."""
    _capture_zipformer_load(monkeypatch)
    monkeypatch.setattr(
        ZipformerEngine, "wheel_supports_cuda", staticmethod(lambda: False)
    )
    engine = ZipformerEngine(model_name="dummy", provider="cuda")

    messages = []
    handler_id = logger.add(lambda m: messages.append(m), level="WARNING")
    try:
        engine.load(warmup_seconds=0)
    finally:
        logger.remove(handler_id)

    assert any("CPU-only" in m for m in messages)


def test_zipformer_does_not_warn_when_the_wheel_matches(monkeypatch):
    _capture_zipformer_load(monkeypatch)
    monkeypatch.setattr(
        ZipformerEngine, "wheel_supports_cuda", staticmethod(lambda: True)
    )
    engine = ZipformerEngine(model_name="dummy", provider="cuda")

    messages = []
    handler_id = logger.add(lambda m: messages.append(m), level="WARNING")
    try:
        engine.load(warmup_seconds=0)
    finally:
        logger.remove(handler_id)

    assert not any("CPU-only" in m for m in messages)


@pytest.mark.parametrize(
    "installed, expected",
    [
        ("1.13.7", False),
        ("1.13.5+cuda12.cudnn9.onnxruntime1.27.1", True),
    ],
)
def test_wheel_supports_cuda_reads_the_installed_local_version(
    monkeypatch, installed, expected
):
    """Both wheels are asserted rather than whichever happens to be installed:
    this runs on dev boxes carrying the CPU wheel and inside the litserver
    image carrying the CUDA one, and must not encode either as "the" answer."""
    monkeypatch.setattr(
        "src.litserver.zipformer.engine.version", lambda _: installed
    )
    assert ZipformerEngine.wheel_supports_cuda() is expected


def test_wheel_supports_cuda_is_false_when_sherpa_onnx_is_absent(monkeypatch):
    def _missing(_):
        raise PackageNotFoundError("sherpa-onnx")

    monkeypatch.setattr(
        "src.litserver.zipformer.engine.version", _missing
    )
    assert ZipformerEngine.wheel_supports_cuda() is False


WORKER_ONLY_PACKAGES = {
    "librosa",
    "numpy",
    "torch",
    "soundfile",
    "sherpa_onnx",
    "litserve",
    "parler_tts",
    "transformers",
}


def _module_path(dotted: str) -> pathlib.Path | None:
    """Where a `src.*` module lives on disk, or None if it is not one."""
    if not dotted.startswith("src"):
        return None
    direct = pathlib.Path(dotted.replace(".", "/") + ".py")
    package = pathlib.Path(dotted.replace(".", "/")) / "__init__.py"
    for candidate in (direct, package):
        if candidate.exists():
            return candidate
    return None


def _imports_of(path: pathlib.Path) -> set[str]:
    import ast

    imported = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    return imported


def test_gateway_import_graph_stays_free_of_worker_dependencies():
    """The gateway image is python:3.12-slim with the base dependencies only.

    Anything main.py can reach at import time must therefore live inside that
    set. Pulling in a worker-only module -- utils/audio.py imports librosa and
    numpy -- does not fail a test run inside the litserver image, where those
    exist; it fails the gateway container at startup with ModuleNotFoundError.
    So this walks the real import graph rather than trusting a run to notice.
    """
    seen, queue, offenders = set(), [pathlib.Path("main.py")], {}
    while queue:
        path = queue.pop()
        if path in seen:
            continue
        seen.add(path)
        for dotted in _imports_of(path):
            root = dotted.split(".")[0]
            if root in WORKER_ONLY_PACKAGES:
                offenders.setdefault(str(path), set()).add(dotted)
            local = _module_path(dotted)
            if local is not None:
                queue.append(local)

    assert not offenders, (
        "gateway import graph reaches worker-only packages: "
        f"{ {k: sorted(v) for k, v in offenders.items()} }"
    )


def test_base_module_does_not_import_torch():
    """The ASR side is pure ONNX. Importing torch in base.py would put a
    multi-GB dependency on that path for one helper only TTS uses."""
    import ast

    tree = ast.parse(pathlib.Path("src/litserver/base.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "torch" not in imported


def test_resolve_device_lives_with_the_engine_that_needs_torch():
    from src.litserver.parler.engine import ParlerTTSEngine

    assert not hasattr(BaseTTSEngine, "resolve_device")
    assert ParlerTTSEngine.resolve_device("cpu") == "cpu"


def test_openai_speech_endpoint_returns_wav(tts_client):
    """Drop-in for a client written against the OpenAI speech API: same body,
    raw audio back, reachable by base URL alone."""
    resp = tts_client.post(
        "/v1/audio/speech",
        json={"input": "হ্যালো", "voice": "Aditi", "response_format": "wav"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    with wave.open(io.BytesIO(resp.content), "rb") as wf:
        assert wf.getframerate() == 44100


def test_openai_speech_pcm_strips_the_wav_header(tts_client):
    """A caller feeding telephony wants frames, not a container."""
    wav = tts_client.post("/v1/audio/speech", json={"input": "হ্যালো"})
    pcm = tts_client.post(
        "/v1/audio/speech", json={"input": "হ্যালো", "response_format": "pcm"}
    )
    assert pcm.status_code == 200
    assert pcm.headers["content-type"] == "audio/pcm"
    assert pcm.headers["X-Audio-Sample-Rate"] == "44100"
    assert len(pcm.content) < len(wav.content)
    assert not pcm.content.startswith(b"RIFF")


def test_speech_stream_returns_pcm_frames(tts_client):
    """stream=true forwards the worker's clauses as they arrive, so the body
    is the concatenated PCM with no container wrapped around it."""
    resp = tts_client.post(
        "/v1/audio/speech",
        json={"input": "হ্যালো", "response_format": "pcm", "stream": True},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/pcm"
    assert resp.headers["X-Audio-Sample-Rate"] == "44100"
    assert not resp.content.startswith(b"RIFF")
    assert len(resp.content) == FAKE_TTS_CHUNKS * FAKE_TTS_CHUNK_FRAMES * 2


def test_speech_stream_and_non_stream_carry_the_same_audio(tts_client):
    """Reassembling the stream must give byte-for-byte what the buffered call
    returns, or the two endpoints are not the same service."""
    streamed = tts_client.post(
        "/v1/audio/speech",
        json={"input": "হ্যালো", "response_format": "pcm", "stream": True},
    )
    buffered = tts_client.post(
        "/v1/audio/speech", json={"input": "হ্যালো", "response_format": "pcm"}
    )
    assert streamed.content == buffered.content


def test_speech_stream_rejects_wav(tts_client):
    """A WAV header states a total length that streaming does not know yet."""
    resp = tts_client.post(
        "/v1/audio/speech",
        json={"input": "হ্যালো", "response_format": "wav", "stream": True},
    )
    assert resp.status_code == 422
    assert "pcm" in resp.json()["detail"]


def test_buffered_wav_covers_every_streamed_chunk(tts_client):
    """The buffered path reassembles all chunks, not just the first."""
    resp = tts_client.post("/v1/audio/speech", json={"input": "হ্যালো"})
    assert resp.status_code == 200
    with wave.open(io.BytesIO(resp.content), "rb") as wf:
        assert wf.getnframes() == FAKE_TTS_CHUNKS * FAKE_TTS_CHUNK_FRAMES


def test_openai_transcriptions_endpoint(asr_client):
    """Drop-in for a client written against the OpenAI transcription API:
    multipart in, {"text": ...} out, segments joined into one utterance."""
    wav = base64.b64decode(make_wav_base64())
    resp = asr_client.post(
        "/v1/audio/transcriptions", files={"file": ("a.wav", wav, "audio/wav")}
    )
    assert resp.status_code == 200
    assert resp.json() == {"text": "হ্যালো"}


def _traces(tmp_path) -> list[Path]:
    return sorted((tmp_path / "traces").glob("*/*"))


def test_asr_route_saves_the_audio_and_transcript_to_the_trace_dir(
    asr_client, tmp_path
):
    """A bad transcript is only debuggable with the clip that produced it, so
    each request leaves the audio as sent and the reply side by side."""
    wav_b64 = make_wav_base64()
    asr_client.post(
        "/asr",
        json={
            "config": {"language": {"sourceLanguage": "bn"}},
            "audio": [{"audioContent": wav_b64}],
        },
    )
    [trace] = _traces(tmp_path)
    assert (trace / "audio_0.wav").read_bytes() == base64.b64decode(wav_b64)
    record = json.loads((trace / "transcript.json").read_text(encoding="utf-8"))
    assert record["route"] == "/asr"
    assert record["status_code"] == 200
    assert record["config"] == {"language": {"sourceLanguage": "bn"}}
    assert record["response"]["output"][0]["source"] == "হ্যালো"


def test_asr_trace_folder_is_named_after_the_first_transcript(asr_client, tmp_path):
    """Named by what was said so a misheard request is found by browsing.
    The fake worker answers "হ্যালো", whose vowel sign and virama must survive
    the filename cleanup intact."""
    body = {
        "config": {"language": {"sourceLanguage": "bn"}},
        "audio": [{"audioContent": make_wav_base64()}],
    }
    asr_client.post("/asr", json=body)
    asr_client.post("/asr", json=body)
    names = [trace.name.split("-", 1)[1] for trace in _traces(tmp_path)]
    assert names[0] == "হ্যালো"
    assert set(names) <= {"হ্যালো", "হ্যালো-2"}


def test_asr_trace_name_falls_back_when_there_is_no_transcript():
    from src.api.v1.asr.trace import _name_from

    assert _name_from({"status_code": 502, "error": "down"}) == "error"
    assert _name_from({"status_code": 200, "response": {"output": []}}) == "empty"
    assert _name_from(
        {"response": {"output": [{"source": "a/b: c?"}]}}
    ) == "a_b_c"


def test_openai_transcriptions_saves_a_trace(asr_client, tmp_path):
    wav = base64.b64decode(make_wav_base64())
    asr_client.post(
        "/v1/audio/transcriptions", files={"file": ("a.wav", wav, "audio/wav")}
    )
    [trace] = _traces(tmp_path)
    assert (trace / "audio_0.wav").read_bytes() == wav
    record = json.loads((trace / "transcript.json").read_text(encoding="utf-8"))
    assert record["config"]["filename"] == "a.wav"


def test_asr_trace_records_failures_too(asr_client, tmp_path, monkeypatch):
    """An unreachable worker is the request someone will most want to replay."""

    async def unreachable(*args, **kwargs):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(
        "src.api.client.transcribe_request", lambda client, payload: unreachable()
    )
    resp = asr_client.post(
        "/asr",
        json={
            "config": {"language": {"sourceLanguage": "bn"}},
            "audio": [{"audioContent": make_wav_base64()}],
        },
    )
    assert resp.status_code == 502
    [trace] = _traces(tmp_path)
    record = json.loads((trace / "transcript.json").read_text(encoding="utf-8"))
    assert record["status_code"] == 502
    assert record["error"] == "LitServe is unreachable"


def test_asr_trace_can_be_disabled(asr_client, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "TRACE_ENABLED", False)
    asr_client.post(
        "/asr",
        json={
            "config": {"language": {"sourceLanguage": "bn"}},
            "audio": [{"audioContent": make_wav_base64()}],
        },
    )
    assert _traces(tmp_path) == []


def test_asr_trace_failure_never_fails_the_request(asr_client, monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("src.api.v1.asr.trace._write", broken)
    resp = asr_client.post(
        "/asr",
        json={
            "config": {"language": {"sourceLanguage": "bn"}},
            "audio": [{"audioContent": make_wav_base64()}],
        },
    )
    assert resp.status_code == 200


def test_asr_trace_prunes_day_folders_past_retention(tmp_path, monkeypatch):
    from datetime import datetime

    from src.api.v1.asr import trace as asr_trace

    root = tmp_path / "traces"
    (root / "2026-01-01").mkdir(parents=True)
    (root / "not-a-date").mkdir()
    monkeypatch.setattr(settings, "TRACE_DIR", str(root))
    monkeypatch.setattr(settings, "TRACE_RETENTION_DAYS", 7)
    monkeypatch.setattr(asr_trace, "_pruned_for", None)

    asr_trace._write("/asr", [b"RIFF"], {}, datetime(2026, 1, 20, 12, 0, 0))

    assert not (root / "2026-01-01").exists()
    assert (root / "not-a-date").exists()
    assert len(list((root / "2026-01-20").iterdir())) == 1


def test_gateway_health_reports_both_models():
    body = TestClient(create_gateway_app()).get("/health").json()
    assert body["status"] == "ok"
    assert body["asr"] == settings.ACTIVE_MODEL_NAME


def test_gateway_sets_cors_headers():
    """The chatbot proxies TTS server-side today only because this service had
    no CORS. With it, a separately hosted UI can call the gateway directly."""
    resp = TestClient(create_gateway_app()).get(
        "/health", headers={"Origin": "https://ui.example.com"}
    )
    assert resp.headers["access-control-allow-origin"] == "*"


def test_asr_route_speaks_the_java_contract(asr_client):
    """The long-standing contract: base64 clips under `audio`, language nested
    under `config.language`, and output[0].source back. Caddy proxies /asr*
    straight here, so it lives on the ASR service itself."""
    resp = asr_client.post(
        "/asr",
        json={
            "config": {"language": {"sourceLanguage": "bn"}},
            "audio": [{"audioContent": make_wav_base64()}],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["output"][0]["source"] == "হ্যালো"


def test_asr_route_forwards_the_body_verbatim(asr_client):
    """Multiple clips and a non-default sourceLanguage must reach LitServe as
    sent, rather than being flattened into one hardcoded bn request."""
    resp = asr_client.post(
        "/asr",
        json={
            "config": {"language": {"sourceLanguage": "en"}},
            "audio": [
                {"audioContent": make_wav_base64()},
                {"audioContent": make_wav_base64()},
            ],
        },
    )
    assert resp.status_code == 200
    assert asr_client.app.state.captured["payload"] == {
        "config": {"language": {"sourceLanguage": "en"}},
        "audio": [
            {"audioContent": make_wav_base64()},
            {"audioContent": make_wav_base64()},
        ],
    }


def test_asr_route_rejects_a_body_without_the_nested_config(asr_client):
    """A flat body is not the contract; it must 422 rather than silently
    transcribing with a default language."""
    resp = asr_client.post("/asr", json={"audio": [{"audioContent": "abc"}]})
    assert resp.status_code == 422


def test_asr_route_no_longer_accepts_a_file_upload(asr_client):
    """The contract carries base64 in the body. Callers holding a file use
    POST /v1/audio/transcriptions instead."""
    wav = base64.b64decode(make_wav_base64())
    resp = asr_client.post(
        "/asr", files={"file": ("recording.webm", wav, "audio/webm")}
    )
    assert resp.status_code == 422


def test_tts_repeat_request_is_served_from_cache(tts_client):
    """The point of the cache: the same text twice occupies the GPU once."""
    body = {"input": "হ্যালো", "tag": "greeting"}
    first = tts_client.post("/v1/audio/speech", json=body)
    second = tts_client.post("/v1/audio/speech", json=body)

    assert first.headers["x-cache"] == "MISS"
    assert second.headers["x-cache"] == "HIT"
    assert second.content == first.content
    assert len(tts_client.synthesize_calls) == 1


def test_tts_cache_is_keyed_on_voice_and_description(tts_client):
    """Same text, different delivery, is different audio and must not share
    an entry -- the failure would be a caller asking for one voice and
    getting whichever one was cached first."""
    tag = {"tag": "greeting"}
    tts_client.post("/v1/audio/speech", json={"input": "হ্যালো"} | tag)
    tts_client.post("/v1/audio/speech", json={"input": "হ্যালো", "voice": "Arjun"} | tag)
    tts_client.post(
        "/v1/audio/speech",
        json={"input": "হ্যালো", "description": "slow and calm"} | tag,
    )
    assert len(tts_client.synthesize_calls) == 3

    repeat = tts_client.post(
        "/v1/audio/speech", json={"input": "হ্যালো", "voice": "Arjun"} | tag
    )
    assert repeat.headers["x-cache"] == "HIT"
    assert len(tts_client.synthesize_calls) == 3


def test_tts_cache_treats_the_default_voice_and_an_explicit_one_as_one_entry(
    tts_client,
):
    """A caller who names the configured default is asking for the audio an
    omitted voice already produced."""
    tts_client.post("/v1/audio/speech", json={"input": "হ্যালো", "tag": "greeting"})
    explicit = tts_client.post(
        "/v1/audio/speech",
        json={"input": "হ্যালো", "voice": settings.TTS_VOICE, "tag": "greeting"},
    )
    assert explicit.headers["x-cache"] == "HIT"
    assert len(tts_client.synthesize_calls) == 1


def test_tts_cache_serves_every_format_from_one_synthesis(tts_client):
    """WAV and PCM are the same audio with and without a 44-byte header, so
    the second format is a rendering decision, not a second GPU call."""
    wav = tts_client.post(
        "/v1/audio/speech", json={"input": "হ্যালো", "tag": "greeting"}
    )
    pcm = tts_client.post(
        "/v1/audio/speech",
        json={"input": "হ্যালো", "response_format": "pcm", "tag": "greeting"},
    )
    assert pcm.headers["x-cache"] == "HIT"
    assert len(tts_client.synthesize_calls) == 1
    assert pcm.content == wav.content[44:]


def test_tts_cache_hit_still_streams_the_same_bytes(tts_client):
    """A hit must be indistinguishable to a streaming client, or code written
    against the streaming contract breaks the moment its text repeats.

    Asserts the byte stream rather than the chunk sizes: raw PCM carries no
    framing, so where the reply is split is the transport's business and no
    caller can depend on it.
    """
    body = {
        "input": "হ্যালো",
        "stream": True,
        "response_format": "pcm",
        "tag": "greeting",
    }
    with tts_client.stream("POST", "/v1/audio/speech", json=body) as live:
        assert live.headers["x-cache"] == "MISS"
        fresh = b"".join(live.iter_raw())
    with tts_client.stream("POST", "/v1/audio/speech", json=body) as replay:
        assert replay.headers["x-cache"] == "HIT"
        cached = b"".join(replay.iter_raw())

    assert len(tts_client.synthesize_calls) == 1
    assert cached == fresh
    assert len(cached) == FAKE_TTS_CHUNKS * FAKE_TTS_CHUNK_FRAMES * 2


def test_tts_streamed_reply_is_cached_for_the_buffered_endpoint(tts_client):
    """One entry serves both modes: a streamed miss fills the cache that a
    later stream=false request reads."""
    with tts_client.stream(
        "POST",
        "/v1/audio/speech",
        json={
            "input": "হ্যালো",
            "stream": True,
            "response_format": "pcm",
            "tag": "greeting",
        },
    ) as live:
        streamed = b"".join(live.iter_raw())

    buffered = tts_client.post(
        "/v1/audio/speech",
        json={"input": "হ্যালো", "response_format": "pcm", "tag": "greeting"},
    )
    assert buffered.headers["x-cache"] == "HIT"
    assert buffered.content == streamed
    assert len(tts_client.synthesize_calls) == 1


def test_tts_cache_can_be_turned_off(tts_client, monkeypatch):
    monkeypatch.setattr(speech_cache, "enabled", False)
    tts_client.post("/v1/audio/speech", json={"input": "হ্যালো"})
    again = tts_client.post("/v1/audio/speech", json={"input": "হ্যালো"})
    assert again.headers["x-cache"] == "MISS"
    assert len(tts_client.synthesize_calls) == 2


def test_tts_cache_rejected_requests_never_reach_the_cache(tts_client):
    """Validation runs ahead of the lookup, so a disabled service cannot
    keep answering from what it cached while it was enabled."""
    tts_client.post("/v1/audio/speech", json={"input": "হ্যালো"})
    settings.TTS_ENABLED = False
    try:
        resp = tts_client.post("/v1/audio/speech", json={"input": "হ্যালো"})
    finally:
        settings.TTS_ENABLED = True
    assert resp.status_code == 503


def test_speech_cache_key_ignores_description_whitespace():
    """The engine strips the description before using it, so " x " and "x"
    produce one recording and must not occupy two entries."""
    cache = SpeechCache(max_bytes=1024)
    assert cache.key("হ্যালো", "Aditi", "  calm  ") == cache.key("হ্যালো", "Aditi", "calm")
    assert cache.key("হ্যালো", "Aditi", None) == cache.key("হ্যালো", "Aditi", "   ")


def test_speech_cache_key_separates_fields_that_could_run_together():
    """Without the separator, text+voice concatenation collides: ("a", "bc")
    and ("ab", "c") would hash the same and swap one caller's audio for
    another's."""
    cache = SpeechCache(max_bytes=1024)
    assert cache.key("a", "bc", None) != cache.key("ab", "c", None)


def test_speech_cache_evicts_the_coldest_entry_to_stay_under_its_cap():
    cache = SpeechCache(max_bytes=200)
    cache.put("a", 44100, [b"\x00" * 100])
    cache.put("b", 44100, [b"\x00" * 100])
    cache.get("a")
    cache.put("c", 44100, [b"\x00" * 100])

    assert cache.get("a") is not None
    assert cache.get("b") is None
    assert cache.get("c") is not None
    assert cache.nbytes <= cache.max_bytes


def test_speech_cache_refuses_an_entry_larger_than_the_whole_cap():
    """Storing it would evict everything else and still not fit, so the one
    oversized reply would empty the cache on every request."""
    cache = SpeechCache(max_bytes=100)
    cache.put("small", 44100, [b"\x00" * 50])
    cache.put("huge", 44100, [b"\x00" * 500])

    assert cache.get("huge") is None
    assert cache.get("small") is not None


def test_speech_cache_replacing_an_entry_does_not_double_count_its_bytes():
    cache = SpeechCache(max_bytes=1000)
    cache.put("a", 44100, [b"\x00" * 100])
    cache.put("a", 44100, [b"\x00" * 100])
    assert cache.nbytes == 100


def test_speech_cache_disabled_stores_nothing():
    cache = SpeechCache(max_bytes=1000, enabled=False)
    cache.put("a", 44100, [b"\x00" * 100])
    assert cache.get("a") is None
    assert cache.nbytes == 0


def test_speech_cache_counts_evictions_so_the_cap_can_be_sized():
    """Evictions are the signal that TTS_CACHE_MAX_MB is too small; without
    the counter, raising it is guesswork."""
    cache = SpeechCache(max_bytes=200)
    cache.put("a", 44100, [b"\x00" * 100])
    cache.put("b", 44100, [b"\x00" * 100])
    assert cache.evictions == 0

    cache.put("c", 44100, [b"\x00" * 100])
    assert cache.evictions == 1
    assert cache.stats()["evictions"] == 1


def test_speech_cache_sizing_says_a_bigger_cap_is_pointless_without_evictions():
    """A cache holding its whole working set cannot be improved with memory,
    and the verdict has to say so rather than invite a bigger number."""
    cache = SpeechCache(max_bytes=1000)
    cache.put("a", 44100, [b"\x00" * 100])
    cache.get("a")
    cache.get("missing")
    assert "buys nothing" in cache.sizing()

    for key in range(20):
        cache.put(str(key), 44100, [b"\x00" * 100])
    assert "may help" in cache.sizing()


def test_speech_cache_clear_resets_every_counter():
    cache = SpeechCache(max_bytes=100)
    cache.put("a", 44100, [b"\x00" * 50])
    cache.get("a")
    cache.get("nope")
    cache.put("huge", 44100, [b"\x00" * 500])
    cache.clear()

    stats = cache.stats()
    assert stats["entries"] == 0
    assert stats["used_mb"] == 0
    assert (stats["hits"], stats["misses"]) == (0, 0)
    assert (stats["evictions"], stats["rejections"]) == (0, 0)


def test_speech_cache_stats_report_mib_not_bytes():
    """The counters exist to be read by a human sizing TTS_CACHE_MAX_MB, and
    268435456 is not a number anyone reads."""
    cache = SpeechCache(max_bytes=256 * 1024 * 1024)
    cache.put("a", 44100, [b"\x00" * (3 * 1024 * 1024)])
    assert cache.stats()["max_mb"] == 256.0
    assert cache.stats()["used_mb"] == 3.0


def test_gateway_health_reports_cache_counters():
    """Sizing the cap is a production decision, so the numbers have to be
    reachable from a deployed box rather than only from a test."""
    body = TestClient(create_gateway_app()).get("/health").json()
    assert body["tts_cache"]["max_mb"] == settings.TTS_CACHE_MAX_MB
    assert "evictions" in body["tts_cache"]


def test_cache_cap_setting_is_mib_converted_to_bytes_once():
    """The knob is MiB for whoever sets it; the cache counts bytes. The
    conversion lives in config so no caller has to remember which unit it
    is holding."""
    assert settings.TTS_CACHE_MAX_BYTES == settings.TTS_CACHE_MAX_MB * 1024 * 1024


def test_tts_only_tagged_replies_are_cached(tts_client):
    """The reason the tag exists: a generated answer is new wording every
    time, so caching it evicts canned answers that are asked constantly."""
    tts_client.post("/v1/audio/speech", json={"input": "উত্তর এক"})
    again = tts_client.post("/v1/audio/speech", json={"input": "উত্তর এক"})

    assert again.headers["x-cache"] == "MISS"
    assert len(tts_client.synthesize_calls) == 2
    assert speech_cache.stats()["entries"] == 0


def test_tts_tagged_reply_is_cached(tts_client):
    body = {"input": "উত্তর এক", "tag": "accepted_nid_types"}
    tts_client.post("/v1/audio/speech", json=body)
    again = tts_client.post("/v1/audio/speech", json=body)

    assert again.headers["x-cache"] == "HIT"
    assert len(tts_client.synthesize_calls) == 1


def test_tts_blank_tag_counts_as_untagged(tts_client):
    """A caller sending "" means the same as sending nothing; trusting them
    to omit the key would make an empty string cache everything."""
    for tag in ("", "   "):
        speech_cache.clear()
        tts_client.synthesize_calls.clear()
        body = {"input": "উত্তর এক", "tag": tag}
        tts_client.post("/v1/audio/speech", json=body)
        again = tts_client.post("/v1/audio/speech", json=body)
        assert again.headers["x-cache"] == "MISS", tag
        assert len(tts_client.synthesize_calls) == 2, tag


def test_tts_tag_changes_nothing_about_which_entry_answers(tts_client):
    """The tag gates the write; the text is the key. Same words under two
    tags is one recording, which is what dedupes a dataset whose tags share
    answers."""
    tts_client.post("/v1/audio/speech", json={"input": "একই কথা", "tag": "tag_a"})
    other = tts_client.post(
        "/v1/audio/speech", json={"input": "একই কথা", "tag": "tag_b"}
    )

    assert other.headers["x-cache"] == "HIT"
    assert len(tts_client.synthesize_calls) == 1
    assert speech_cache.stats()["entries"] == 1


def test_tts_untagged_request_may_still_read_a_cached_reply(tts_client):
    """Gating the write but not the read: the key is the text, so a hit is
    the right audio for whoever asks. Refusing it would re-synthesize
    something already in hand."""
    tts_client.post("/v1/audio/speech", json={"input": "একই কথা", "tag": "tag_a"})
    untagged = tts_client.post("/v1/audio/speech", json={"input": "একই কথা"})

    assert untagged.headers["x-cache"] == "HIT"
    assert len(tts_client.synthesize_calls) == 1


def test_tts_edited_answer_under_one_tag_never_serves_the_old_audio(tts_client):
    """The case that decided the design: the dataset is fees and dates on a
    refresh timer, so the text behind a tag changes with no deploy. Keying on
    the tag would keep speaking the old number."""
    fee = "ফি দুইশ ত্রিশ টাকা"
    tts_client.post("/v1/audio/speech", json={"input": fee, "tag": "card_fees"})

    edited = "ফি দুইশ পঞ্চাশ টাকা"
    resp = tts_client.post(
        "/v1/audio/speech", json={"input": edited, "tag": "card_fees"}
    )

    assert resp.headers["x-cache"] == "MISS"
    assert len(tts_client.synthesize_calls) == 2


def test_tts_tagged_streamed_reply_is_cached_but_untagged_is_not(tts_client):
    """The gate has to hold on the streaming path too, where the store
    happens inside the response generator."""
    def stream(body):
        with tts_client.stream("POST", "/v1/audio/speech", json=body) as resp:
            return b"".join(resp.iter_raw()), resp.headers["x-cache"]

    plain = {"input": "উত্তর এক", "stream": True, "response_format": "pcm"}
    stream(plain)
    assert stream(plain)[1] == "MISS"
    assert speech_cache.stats()["entries"] == 0

    tagged = plain | {"tag": "accepted_nid_types"}
    stream(tagged)
    assert stream(tagged)[1] == "HIT"
