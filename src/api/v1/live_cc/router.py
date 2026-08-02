import base64
import io
import math
import wave

import httpx
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

from src.api import client as litserve_client
from src.core.config import settings

router = APIRouter(prefix="/v1/live-cc", tags=["Live CC"])

_PCM_SAMPLE_WIDTH_BYTES = 2


def _pcm_to_wav_bytes(pcm_bytes: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(_PCM_SAMPLE_WIDTH_BYTES)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


async def _transcribe(
    client: httpx.AsyncClient, pcm_bytes: bytes, input_sr: int
) -> str:
    wav_bytes = _pcm_to_wav_bytes(pcm_bytes, input_sr)
    audio_content_b64 = base64.b64encode(wav_bytes).decode("utf-8")
    resp = await litserve_client.transcribe(client, audio_content_b64)
    data = resp.json()
    return " ".join(item.get("source", "") for item in data.get("output", []))


async def _embed(
    client: httpx.AsyncClient, pcm_bytes: bytes, input_sr: int
) -> list[float]:
    """Call LitServe's internal speaker-embedding endpoint for one PCM segment."""
    wav_bytes = _pcm_to_wav_bytes(pcm_bytes, input_sr)
    audio_content_b64 = base64.b64encode(wav_bytes).decode("utf-8")
    return await litserve_client.embed(client, audio_content_b64)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def _is_target_speaker(
    client: httpx.AsyncClient,
    pcm_bytes: bytes,
    input_sr: int,
    enrollment_embedding: list[float],
) -> bool:
    embedding = await _embed(client, pcm_bytes, input_sr)
    similarity = _cosine_similarity(embedding, enrollment_embedding)
    return similarity >= settings.SPEAKER_SIMILARITY_THRESHOLD


@router.websocket("/ws")
async def live_cc_ws(websocket: WebSocket) -> None:
    """Live closed-captioning: client streams raw 16-bit PCM mono audio at
    settings.LIVE_CC_INPUT_SAMPLE_RATE. Every LIVE_CC_INTERIM_INTERVAL_SECONDS
    of in-progress audio, the whole in-progress chunk is re-transcribed and
    pushed as an interim caption (is_final: false) — gives the feel of
    live-updating text without incremental decoding. Every LIVE_CC_CHUNK_SECONDS
    the chunk is finalized (is_final: true) and the buffer resets.

    Each chunk is labeled with the *input* rate, not the model's rate —
    decode_base64_audio (via the /predict re-dispatch below) resamples from
    whatever the WAV header says to settings.SAMPLE_RATE, same as any other
    upload. Mislabeling this would silently corrupt the audio (wrong
    playback speed) rather than fail loudly.

    Reuses the same model as batch requests by re-dispatching each buffered
    chunk via src.api.client as a real HTTP request against the LitServe
    model server (same pattern as /v1/asr/transcribe/file) — this process
    never holds the model itself, that lives in LitServe's own worker
    processes.

    Optional speaker gate (SPEAKER_GATE_ENABLED, off by default): the first
    SPEAKER_ENROLL_SECONDS of the call enroll a reference voice embedding
    (via LitServe's /internal/speaker/embed), assuming the target caller is
    the first person heard — a background voice speaking first would enroll
    the wrong speaker. Every chunk after that is embedded and compared to the
    enrollment; chunks below SPEAKER_SIMILARITY_THRESHOLD similarity are
    dropped (no caption emitted at all) instead of transcribed. This is
    segment-level gating only — it can't separate two voices talking at the
    same time within one chunk, only decide whether a whole chunk sounds like
    the enrolled speaker or not.
    """
    await websocket.accept()

    input_sr = settings.LIVE_CC_INPUT_SAMPLE_RATE
    chunk_byte_target = (
        int(input_sr * settings.LIVE_CC_CHUNK_SECONDS) * _PCM_SAMPLE_WIDTH_BYTES
    )
    interim_byte_step = (
        int(input_sr * settings.LIVE_CC_INTERIM_INTERVAL_SECONDS)
        * _PCM_SAMPLE_WIDTH_BYTES
    )
    enroll_byte_target = (
        int(input_sr * settings.SPEAKER_ENROLL_SECONDS) * _PCM_SAMPLE_WIDTH_BYTES
    )

    buffer = bytearray()
    next_interim_at = interim_byte_step

    gate_enabled = settings.SPEAKER_GATE_ENABLED
    enrollment_buffer = bytearray()
    enrollment_embedding: list[float] | None = None

    async with litserve_client.get_litserve_client() as client:
        try:
            while True:
                chunk = await websocket.receive_bytes()
                buffer.extend(chunk)

                if gate_enabled and enrollment_embedding is None:
                    enrollment_buffer.extend(chunk)
                    if len(enrollment_buffer) >= enroll_byte_target:
                        try:
                            enrollment_embedding = await _embed(
                                client, bytes(enrollment_buffer), input_sr
                            )
                        except Exception as exc:
                            logger.warning(
                                "Speaker enrollment failed, retrying with "
                                f"more audio: {exc}"
                            )

                while len(buffer) >= chunk_byte_target:
                    segment = bytes(buffer[:chunk_byte_target])
                    del buffer[:chunk_byte_target]
                    if enrollment_embedding is None or await _is_target_speaker(
                        client, segment, input_sr, enrollment_embedding
                    ):
                        text = await _transcribe(client, segment, input_sr)
                        await websocket.send_json(
                            {"text": text, "is_final": True}
                        )
                    next_interim_at = interim_byte_step

                if interim_byte_step > 0 and len(buffer) >= next_interim_at:
                    segment = bytes(buffer)
                    if enrollment_embedding is None or await _is_target_speaker(
                        client, segment, input_sr, enrollment_embedding
                    ):
                        text = await _transcribe(client, segment, input_sr)
                        await websocket.send_json(
                            {"text": text, "is_final": False}
                        )
                    next_interim_at += interim_byte_step
        except WebSocketDisconnect:
            pass
