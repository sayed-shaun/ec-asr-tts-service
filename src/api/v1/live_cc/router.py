import base64
import io
import wave

import httpx
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.core.config import settings

router = APIRouter(prefix="/v1/live-cc", tags=["live-cc"])

_PCM_SAMPLE_WIDTH_BYTES = 2  # 16-bit PCM


def _pcm_to_wav_bytes(pcm_bytes: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(_PCM_SAMPLE_WIDTH_BYTES)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


@router.websocket("/ws")
async def live_cc_ws(websocket: WebSocket) -> None:
    """Live closed-captioning: client streams raw 16-bit PCM mono audio at
    settings.SAMPLE_RATE; every LIVE_CC_CHUNK_SECONDS worth of audio is
    transcribed and the resulting caption text is pushed back.

    Reuses the same model as batch requests by re-dispatching each buffered
    chunk to settings.API_PATH over an in-process ASGI call (same pattern as
    /v1/asr/transcribe/file) — this process never holds the model itself,
    that lives in LitServe's own worker processes.
    """
    await websocket.accept()

    chunk_byte_target = int(settings.SAMPLE_RATE * settings.LIVE_CC_CHUNK_SECONDS) * _PCM_SAMPLE_WIDTH_BYTES
    buffer = bytearray()

    transport = httpx.ASGITransport(app=websocket.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://internal") as client:
        try:
            while True:
                buffer.extend(await websocket.receive_bytes())

                while len(buffer) >= chunk_byte_target:
                    segment = bytes(buffer[:chunk_byte_target])
                    del buffer[:chunk_byte_target]

                    wav_bytes = _pcm_to_wav_bytes(segment, settings.SAMPLE_RATE)
                    payload = {
                        "config": {"language": {"sourceLanguage": "bn"}},
                        "audio": [{"audioContent": base64.b64encode(wav_bytes).decode("utf-8")}],
                    }
                    resp = await client.post(settings.API_PATH, json=payload, timeout=120)
                    data = resp.json()
                    text = " ".join(item.get("source", "") for item in data.get("output", []))
                    await websocket.send_json({"text": text, "is_final": True})
        except WebSocketDisconnect:
            pass
