"""Streaming WebSocket service for live voicebot/callbot use.

Owns the engines in-process, unlike the gateway. That is forced rather than
chosen: a live call needs one ASR stream held open across frames and a TTS
generation that can be cancelled mid-sentence, and LitServe has no WebSocket
support and runs stateless request workers in separate processes, so neither
fits behind it. The LitServe path stays the right one for batch and REST.

Two sockets rather than one duplex socket: the LLM that joins them is not part
of this repo, so the orchestrator owns turn-taking and drives both.
"""

import asyncio
import json
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from loguru import logger

from src.core.config import settings
from src.litserver.parler import engine as parler
from src.litserver.parler.chunking import IncrementalTextChunker
from src.litserver.parler.voices import VOICES
from src.litserver.zipformer import engine as zipformer

PCM_SCALE = 32768.0


def pcm_to_float32(payload: bytes) -> np.ndarray:
    """Decode signed 16-bit little-endian PCM, the wire format telephony sends."""
    return np.frombuffer(payload, dtype="<i2").astype(np.float32) / PCM_SCALE


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load both models once, before the first call connects.

    Zipformer is the ASR engine: it decodes incrementally, which is what a
    live call needs and what whole-utterance engines cannot give.
    """
    app.state.asr = zipformer.build("cpu")
    await asyncio.to_thread(app.state.asr.load, sample_rate=settings.SAMPLE_RATE)

    app.state.tts = None
    app.state.tts_lock = asyncio.Lock()
    if settings.TTS_ENABLED:
        app.state.tts = parler.build(settings.ACCELERATOR)
        await asyncio.to_thread(app.state.tts.load)
    yield


def create_voicebot_app() -> FastAPI:
    app = FastAPI(title="Bangla voicebot streaming", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "asr": settings.ZIPFORMER_MODEL_NAME,
            "tts": settings.ACTIVE_TTS_MODEL_NAME,
        }

    @app.websocket("/api/v1/voicebot/asr")
    async def asr_ws(websocket: WebSocket) -> None:
        """Caller streams 16-bit PCM; server answers with partial and final text.

        One ZipformerSession lives for the whole connection, so decoding is
        incremental. A final is emitted whenever the recognizer's endpoint
        rules say the caller stopped talking, and the session resets for the
        next turn without tearing down the stream.

        Concurrency: every connection gets its own sherpa stream off the shared
        recognizer, which is what sherpa-onnx's multi-stream design expects, so
        calls decode independently rather than queueing behind one lock.
        """
        await websocket.accept()
        session = websocket.app.state.asr.stream()
        rate = settings.VOICEBOT_INPUT_SAMPLE_RATE
        await websocket.send_json(
            {"type": "ready", "sample_rate": rate, "format": "pcm_s16le"}
        )

        last_partial = ""
        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    return

                if (payload := message.get("bytes")) is not None:
                    text = await asyncio.to_thread(
                        session.accept, pcm_to_float32(payload), rate
                    )
                    if await asyncio.to_thread(session.is_endpoint):
                        if text.strip():
                            await websocket.send_json({"type": "final", "text": text})
                        session.reset()
                        last_partial = ""
                    elif text != last_partial:
                        await websocket.send_json({"type": "partial", "text": text})
                        last_partial = text
                    continue

                if (raw := message.get("text")) is not None:
                    if json.loads(raw).get("type") == "end":
                        text = await asyncio.to_thread(session.finish, 1.0, rate)
                        if text.strip():
                            await websocket.send_json({"type": "final", "text": text})
                        await websocket.send_json({"type": "done"})
                        return
        except WebSocketDisconnect:
            return

    @app.websocket("/api/v1/voicebot/tts")
    async def tts_ws(websocket: WebSocket) -> None:
        """Server streams PCM as each clause finishes, and can be interrupted.

        Text arrives as deltas so an LLM's output can be fed in as it is
        generated; IncrementalTextChunker releases a clause as soon as one is
        complete. "cancel" is barge-in: it bumps the generation counter, which
        both drops everything queued and stops the frames of whatever is
        mid-send, so the caller stops hearing the old answer immediately.
        """
        await websocket.accept()
        engine = websocket.app.state.tts
        if engine is None:
            await websocket.close(code=1013, reason="TTS is disabled")
            return

        lock = websocket.app.state.tts_lock
        frame_ms = max(20, settings.VOICEBOT_PCM_FRAME_MS)
        chunker = IncrementalTextChunker(max_chars=settings.TTS_MAX_CHARS)
        queue: asyncio.Queue = asyncio.Queue()
        send_lock = asyncio.Lock()
        state = {"generation": 0, "sequence": 0}
        voice = {"name": settings.TTS_VOICE, "description": None}

        async def send_json(payload: dict[str, Any]) -> None:
            async with send_lock:
                await websocket.send_json(payload)

        async def enqueue(chunks: list[str]) -> None:
            for text in chunks:
                await queue.put((state["generation"], state["sequence"], text))
                state["sequence"] += 1

        async def discard_queued() -> None:
            while not queue.empty():
                queue.get_nowait()
                queue.task_done()

        async def worker() -> None:
            while True:
                item = await queue.get()
                try:
                    if item is None:
                        return
                    generation, sequence, text = item
                    if generation != state["generation"]:
                        continue
                    started = time.perf_counter()
                    async with lock:
                        audio = await asyncio.to_thread(
                            engine.synthesize, text, voice["name"], voice["description"]
                        )
                    if generation != state["generation"]:
                        continue

                    pcm = audio.pcm_s16le()
                    frame_bytes = max(2, audio.sample_rate * frame_ms // 1000 * 2)
                    await send_json(
                        {
                            "type": "audio_start",
                            "generation": generation,
                            "sequence": sequence,
                            "text": text,
                            "sample_rate": audio.sample_rate,
                            "bytes": len(pcm),
                            "generation_ms": round(
                                (time.perf_counter() - started) * 1000
                            ),
                        }
                    )
                    for offset in range(0, len(pcm), frame_bytes):
                        if generation != state["generation"]:
                            break
                        async with send_lock:
                            await websocket.send_bytes(
                                pcm[offset : offset + frame_bytes]
                            )
                    if generation == state["generation"]:
                        await send_json(
                            {
                                "type": "audio_end",
                                "generation": generation,
                                "sequence": sequence,
                            }
                        )
                except Exception as exc:
                    logger.warning(f"TTS synthesis failed: {exc}")
                    await send_json({"type": "error", "message": str(exc)})
                finally:
                    queue.task_done()

        task = asyncio.create_task(worker())
        await send_json(
            {
                "type": "ready",
                "format": "pcm_s16le",
                "channels": 1,
                "voices": sorted(VOICES),
                "default_voice": settings.TTS_VOICE,
            }
        )

        try:
            while True:
                message = json.loads(await websocket.receive_text())
                event = message.get("type")

                if event == "configure":
                    name = message.get("voice", voice["name"])
                    if name not in VOICES:
                        await send_json(
                            {"type": "error", "message": f"unknown voice: {name}"}
                        )
                        continue
                    voice["name"] = name
                    voice["description"] = message.get("description")
                    await send_json({"type": "configured", "voice": name})
                elif event == "text":
                    await enqueue(chunker.feed(message.get("text", "")))
                elif event == "flush":
                    await enqueue(chunker.flush())
                elif event == "cancel":
                    state["generation"] += 1
                    state["sequence"] = 0
                    chunker.clear()
                    await discard_queued()
                    await send_json(
                        {"type": "cancelled", "generation": state["generation"]}
                    )
                elif event == "end":
                    await enqueue(chunker.flush())
                    await queue.join()
                    await send_json({"type": "done"})
                    await queue.put(None)
                    await task
                    return
                else:
                    await send_json(
                        {
                            "type": "error",
                            "message": (
                                "type must be configure, text, flush, "
                                "cancel or end"
                            ),
                        }
                    )
        except WebSocketDisconnect:
            state["generation"] += 1
        finally:
            if not task.done():
                task.cancel()

    return app
