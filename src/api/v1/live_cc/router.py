import base64
import io
import wave

import httpx
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.api import client as litserve_client
from src.core.config import settings

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


@router.websocket("/ws")
async def live_cc_ws(websocket: WebSocket) -> None:
    """Client streams raw 16-bit PCM at LIVE_CC_INPUT_SAMPLE_RATE. Every
    LIVE_CC_INTERIM_INTERVAL_SECONDS the whole in-progress chunk is
    re-transcribed as an interim caption (is_final: false) — fakes
    incremental decoding without real streaming ASR. Every
    LIVE_CC_CHUNK_SECONDS the chunk finalizes (is_final: true) and resets.

    Chunks must be labeled with the input rate, not SAMPLE_RATE — mislabeling
    silently corrupts audio (wrong playback speed) rather than failing loudly.
    """
    await websocket.accept()

    input_sr = settings.LIVE_CC_INPUT_SAMPLE_RATE
    chunk_byte_target = seconds_to_bytes(settings.LIVE_CC_CHUNK_SECONDS, input_sr)
    interim_byte_step = seconds_to_bytes(
        settings.LIVE_CC_INTERIM_INTERVAL_SECONDS, input_sr
    )

    buffer = bytearray()
    next_interim_at = interim_byte_step

    async with litserve_client.get_litserve_client() as client:

        async def emit_caption(segment: bytes, is_final: bool) -> None:
            text = await transcribe_pcm(client, segment, input_sr)
            await websocket.send_json({"text": text, "is_final": is_final})

        try:
            while True:
                chunk = await websocket.receive_bytes()
                buffer.extend(chunk)

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
