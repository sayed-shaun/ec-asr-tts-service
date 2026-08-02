import base64
import io
import wave

import httpx
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

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


async def _transcribe(client: httpx.AsyncClient, pcm_bytes: bytes, input_sr: int) -> str:
    wav_bytes = _pcm_to_wav_bytes(pcm_bytes, input_sr)
    payload = {
        "config": {"language": {"sourceLanguage": "bn"}},
        "audio": [{"audioContent": base64.b64encode(wav_bytes).decode("utf-8")}],
    }
    resp = await client.post(settings.API_PATH, json=payload, timeout=120)
    data = resp.json()
    return " ".join(item.get("source", "") for item in data.get("output", []))


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
    chunk to settings.API_PATH as a real HTTP request against the LitServe
    model server (same pattern as /v1/asr/transcribe/file) — this process
    never holds the model itself, that lives in LitServe's own worker
    processes.
    """
    await websocket.accept()

    input_sr = settings.LIVE_CC_INPUT_SAMPLE_RATE
    chunk_byte_target = int(input_sr * settings.LIVE_CC_CHUNK_SECONDS) * _PCM_SAMPLE_WIDTH_BYTES
    interim_byte_step = int(input_sr * settings.LIVE_CC_INTERIM_INTERVAL_SECONDS) * _PCM_SAMPLE_WIDTH_BYTES

    buffer = bytearray()
    next_interim_at = interim_byte_step

    litserve_base_url = f"http://{settings.LITSERVE_HOST}:{settings.LITSERVE_PORT}"
    async with httpx.AsyncClient(base_url=litserve_base_url) as client:
        try:
            while True:
                buffer.extend(await websocket.receive_bytes())

                while len(buffer) >= chunk_byte_target:
                    segment = bytes(buffer[:chunk_byte_target])
                    del buffer[:chunk_byte_target]
                    text = await _transcribe(client, segment, input_sr)
                    await websocket.send_json({"text": text, "is_final": True})
                    next_interim_at = interim_byte_step

                if interim_byte_step > 0 and len(buffer) >= next_interim_at:
                    text = await _transcribe(client, bytes(buffer), input_sr)
                    await websocket.send_json({"text": text, "is_final": False})
                    next_interim_at += interim_byte_step
        except WebSocketDisconnect:
            pass
