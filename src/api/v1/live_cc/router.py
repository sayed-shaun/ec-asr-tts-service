import base64
import io
import wave

import httpx
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

from src.api import client as litserve_client
from src.core.config import settings
from src.utils.metrics import cosine_similarity

router = APIRouter(prefix="/api/v1/live-cc", tags=["Live CC"])

PCM_SAMPLE_WIDTH_BYTES = 2


def seconds_to_bytes(seconds: float, sample_rate: int) -> int:
    return int(sample_rate * seconds) * PCM_SAMPLE_WIDTH_BYTES


def pcm_to_wav_b64(pcm_bytes: bytes, sample_rate: int) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(PCM_SAMPLE_WIDTH_BYTES)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


async def transcribe_pcm(
    client: httpx.AsyncClient, pcm_bytes: bytes, input_sr: int
) -> str:
    resp = await litserve_client.transcribe(client, pcm_to_wav_b64(pcm_bytes, input_sr))
    data = resp.json()
    return " ".join(item.get("source", "") for item in data.get("output", []))


async def embed_pcm(
    client: httpx.AsyncClient, pcm_bytes: bytes, input_sr: int
) -> list[float]:
    """Call LitServe's internal speaker-embedding endpoint for one PCM segment."""
    return await litserve_client.embed(client, pcm_to_wav_b64(pcm_bytes, input_sr))


async def is_target_speaker(
    client: httpx.AsyncClient,
    pcm_bytes: bytes,
    input_sr: int,
    enrollment_embedding: list[float],
) -> bool:
    embedding = await embed_pcm(client, pcm_bytes, input_sr)
    similarity = cosine_similarity(embedding, enrollment_embedding)
    return similarity >= settings.SPEAKER_SIMILARITY_THRESHOLD


@router.websocket("/ws")
async def live_cc_ws(websocket: WebSocket) -> None:
    """Client streams raw 16-bit PCM at LIVE_CC_INPUT_SAMPLE_RATE. Every
    LIVE_CC_INTERIM_INTERVAL_SECONDS the whole in-progress chunk is
    re-transcribed as an interim caption (is_final: false) — fakes
    incremental decoding without real streaming ASR. Every
    LIVE_CC_CHUNK_SECONDS the chunk finalizes (is_final: true) and resets.

    Chunks must be labeled with the input rate, not SAMPLE_RATE — mislabeling
    silently corrupts audio (wrong playback speed) rather than failing loudly.

    If SPEAKER_GATE_ENABLED: the first SPEAKER_ENROLL_SECONDS enroll a
    reference voice embedding (assumes the target caller speaks first).
    Later chunks below SPEAKER_SIMILARITY_THRESHOLD similarity to that
    embedding are dropped silently. Segment-level only — can't separate
    two people talking at once within one chunk.
    """
    await websocket.accept()

    input_sr = settings.LIVE_CC_INPUT_SAMPLE_RATE
    chunk_byte_target = seconds_to_bytes(settings.LIVE_CC_CHUNK_SECONDS, input_sr)
    interim_byte_step = seconds_to_bytes(
        settings.LIVE_CC_INTERIM_INTERVAL_SECONDS, input_sr
    )
    enroll_byte_target = seconds_to_bytes(settings.SPEAKER_ENROLL_SECONDS, input_sr)

    buffer = bytearray()
    next_interim_at = interim_byte_step

    gate_enabled = settings.SPEAKER_GATE_ENABLED
    enrollment_buffer = bytearray()
    enrollment_embedding: list[float] | None = None

    async with litserve_client.get_litserve_client() as client:

        async def emit_caption(segment: bytes, is_final: bool) -> None:
            if enrollment_embedding is not None and not await is_target_speaker(
                client, segment, input_sr, enrollment_embedding
            ):
                return
            text = await transcribe_pcm(client, segment, input_sr)
            await websocket.send_json({"text": text, "is_final": is_final})

        try:
            while True:
                chunk = await websocket.receive_bytes()
                buffer.extend(chunk)

                if gate_enabled and enrollment_embedding is None:
                    enrollment_buffer.extend(chunk)
                    if len(enrollment_buffer) >= enroll_byte_target:
                        try:
                            enrollment_embedding = await embed_pcm(
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
                    await emit_caption(segment, is_final=True)
                    next_interim_at = interim_byte_step

                if interim_byte_step > 0 and len(buffer) >= next_interim_at:
                    await emit_caption(bytes(buffer), is_final=False)
                    next_interim_at += interim_byte_step
        except WebSocketDisconnect:
            pass
