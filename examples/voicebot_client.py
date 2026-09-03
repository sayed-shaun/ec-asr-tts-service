"""Drive the voicebot WebSockets from the command line.

    python examples/voicebot_client.py asr path/to/audio.wav
    python examples/voicebot_client.py tts "আমি ভালো আছি।"

ASR streams the file in as PCM frames the way telephony would and prints
partial and final transcripts. TTS sends text, collects the streamed PCM and
writes it to voicebot_tts.wav.
"""

import asyncio
import json
import sys
import wave

import numpy as np
import soundfile as sf
import websockets

HOST = "ws://localhost:8100/api/v1/voicebot"
SR = 16000
FRAME_MS = 100


async def run_asr(path: str) -> None:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")

    frame = int(SR * FRAME_MS / 1000)
    async with websockets.connect(f"{HOST}/asr") as ws:
        print(json.loads(await ws.recv()))

        async def read() -> None:
            try:
                async for message in ws:
                    event = json.loads(message)
                    kind = event.get("type")
                    if kind in ("partial", "final"):
                        print(f"  {kind:8} {event.get('text', '')}")
                    else:
                        print(f"  {kind}")
                    if kind == "done":
                        return
            except websockets.ConnectionClosed:
                pass

        reader = asyncio.create_task(read())
        for offset in range(0, len(pcm), frame):
            await ws.send(pcm[offset : offset + frame].tobytes())
            await asyncio.sleep(FRAME_MS / 1000)
        await ws.send(json.dumps({"type": "end"}))
        await reader


async def run_tts(text: str) -> None:
    chunks: list[bytes] = []
    rate = 44100
    async with websockets.connect(f"{HOST}/tts") as ws:
        ready = json.loads(await ws.recv())
        print(ready)
        await ws.send(json.dumps({"type": "text", "text": text}))
        await ws.send(json.dumps({"type": "flush"}))
        await ws.send(json.dumps({"type": "end"}))
        async for message in ws:
            if isinstance(message, bytes):
                chunks.append(message)
                continue
            event = json.loads(message)
            print(" ", event)
            if event.get("type") == "audio_start":
                rate = event.get("sample_rate", rate)
            if event.get("type") == "done":
                break

    with wave.open("voicebot_tts.wav", "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(b"".join(chunks))
    print(f"wrote voicebot_tts.wav ({sum(map(len, chunks))} bytes @ {rate}Hz)")


if __name__ == "__main__":
    mode, arg = sys.argv[1], sys.argv[2]
    asyncio.run(run_asr(arg) if mode == "asr" else run_tts(arg))
