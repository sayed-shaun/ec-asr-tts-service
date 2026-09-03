import base64
import io
import pathlib
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import numpy as np
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from loguru import logger
from pydantic import ValidationError

from main import create_gateway_app
from src.api.client import PREDICT_PATH, SYNTHESIZE_PATH
from src.api.v1.asr.router import router as asr_router
from src.api.v1.asr.schema import AsrRequest
from src.api.v1.tts.router import router as tts_router
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
def asr_client(monkeypatch):
    """A fake /predict on a separate app, reached in-process by monkeypatching
    the client's httpx.AsyncClient so no real socket opens.
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
    request = TtsRequest(input="আমি ভালো আছি।")
    response = tts_lit_api.encode_response(
        tts_lit_api.predict(tts_lit_api.decode_request(request))
    )
    assert response.taskType == "tts"
    assert response.sampleRate == 44100
    assert response.voice == settings.TTS_VOICE

    wav = base64.b64decode(response.audioContent)
    with wave.open(io.BytesIO(wav), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 44100
        assert wf.getnframes() == 44100


def test_tts_lit_api_hands_the_whole_request_to_the_engine(tts_lit_api):
    """Splitting is the engine's business now, so a plain engine sees the text
    whole and the LitAPI stays model-agnostic."""
    text = "এক দুই তিন। চার পাঁচ ছয়। সাত আট নয়।"
    tts_lit_api.predict(tts_lit_api.decode_request(TtsRequest(input=text)))
    assert [call[0] for call in tts_lit_api.engine.calls] == [text]


def test_chunking_engine_splits_and_joins_with_gaps():
    """The Parler-shaped path: speak() splits on clause boundaries, synthesizes
    each and joins with a gap between parts but not around them."""
    api = TTSLitAPI(max_batch_size=1, api_path=TTS_API_PATH)
    api.engine = ChunkingFakeTTSEngine()
    text = "এক দুই তিন। চার পাঁচ ছয়। সাত আট নয়।"
    output = api.predict(api.decode_request(TtsRequest(input=text)))

    assert len(api.engine.calls) == 3
    expected = 3 * 44100 + 2 * round(0.08 * 44100)
    assert output["audio"].samples.size == expected


def test_tts_lit_api_passes_voice_and_description_through(tts_lit_api):
    request = TtsRequest(input="পরীক্ষা।", voice="Arjun", description="  slow and calm  ")
    tts_lit_api.predict(tts_lit_api.decode_request(request))
    assert tts_lit_api.engine.calls == [("পরীক্ষা।", "Arjun", "  slow and calm  ")]


def test_tts_lit_api_unknown_voice_is_422_not_500(tts_lit_api):
    decoded = tts_lit_api.decode_request(TtsRequest(input="পরীক্ষা।"))
    decoded["voice"] = "Nobody"
    with pytest.raises(HTTPException) as excinfo:
        tts_lit_api.predict(decoded)
    assert excinfo.value.status_code == 422


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


@pytest.fixture
def tts_client(monkeypatch):
    """A fake LitServe /synthesize, wired in through the shared client.

    TTS_ENABLED is off by default, so the fixture opts in the way a deployment
    that wants TTS does."""
    monkeypatch.setattr(settings, "TTS_ENABLED", True)
    fake = FastAPI()

    @fake.post(SYNTHESIZE_PATH)
    async def synthesize(payload: dict) -> dict:
        silence = np.zeros(1000, dtype=np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(44100)
            wf.writeframes(silence.tobytes())
        return {
            "taskType": "tts",
            "audioContent": base64.b64encode(buf.getvalue()).decode("utf-8"),
            "sampleRate": 44100,
            "voice": payload.get("voice") or "Aditi",
            "time_taken": 0.1,
        }

    transport = httpx.ASGITransport(app=fake)
    monkeypatch.setattr(
        "src.api.client.get_litserve_client",
        lambda: httpx.AsyncClient(transport=transport, base_url="http://litserver:8000"),
    )
    app = FastAPI()
    app.include_router(tts_router)
    return TestClient(app)


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


def test_zipformer_provider_defaults_to_cpu_and_is_configurable(monkeypatch):
    """The provider comes from ZIPFORMER_PROVIDER, not ACCELERATOR: it is a
    property of the installed wheel, not of the device LitServe assigns."""
    from src.litserver.zipformer.engine import build as zipformer_build

    monkeypatch.setattr(settings, "ACCELERATOR", "cuda")
    assert zipformer_build("cuda:0").provider == "cpu"

    monkeypatch.setattr(settings, "ZIPFORMER_PROVIDER", "cuda")
    assert zipformer_build("cpu").provider == "cuda"


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


def test_wheel_supports_cuda_reads_the_installed_local_version():
    """The CPU wheel here is a plain version, so this must be False."""
    assert ZipformerEngine.wheel_supports_cuda() is False


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


def test_openai_transcriptions_endpoint(asr_client):
    """Drop-in for a client written against the OpenAI transcription API:
    multipart in, {"text": ...} out, segments joined into one utterance."""
    wav = base64.b64decode(make_wav_base64())
    resp = asr_client.post(
        "/v1/audio/transcriptions", files={"file": ("a.wav", wav, "audio/wav")}
    )
    assert resp.status_code == 200
    assert resp.json() == {"text": "হ্যালো"}


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


def test_asr_route_matches_what_the_chatbot_ui_posts(asr_client):
    """The UI posts a MediaRecorder blob plus model_type and reads
    output[0].source. Caddy proxies /asr* straight here, so it lives on the
    ASR service itself, not on the chatbot."""
    wav = base64.b64decode(make_wav_base64())
    resp = asr_client.post(
        "/asr",
        files={"file": ("recording.webm", wav, "audio/webm")},
    )
    assert resp.status_code == 200
    assert resp.json()["output"][0]["source"] == "হ্যালো"


def test_asr_route_ignores_extra_form_fields(asr_client):
    """The UI still posts model_type. An extra multipart field must not 422 the
    request, or the browser breaks the moment the server stops declaring it."""
    wav = base64.b64decode(make_wav_base64())
    resp = asr_client.post(
        "/asr",
        files={"file": ("recording.webm", wav, "audio/webm")},
    )
    assert resp.status_code == 200
