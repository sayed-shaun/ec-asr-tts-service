import base64
import binascii
import io
import json
import wave
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import Response, StreamingResponse

from src.api import client as litserve_client
from src.api.v1.tts.schema import TtsRequest
from src.core.config import settings
from src.litserver.parler.voices import VOICES

router = APIRouter(tags=["TTS"])


def wav_from_pcm(pcm: bytes, sample_rate: int) -> bytes:
    """Wrap already-encoded 16-bit mono PCM in a WAV container.

    The streaming path carries raw PCM per chunk, because concatenating WAV
    files would splice a 44-byte header into the middle of the audio. This
    reassembles the single header once the whole reply is in hand.

    Stdlib only, and here rather than in utils/audio.py on purpose: that
    module imports librosa and numpy, which the gateway image deliberately
    does not install. Importing it from this router puts a multi-GB decoding
    stack on the gateway's import path and the container dies at startup with
    ModuleNotFoundError.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return buf.getvalue()


def require_tts_enabled() -> None:
    """503 rather than 404 when TTS_ENABLED is false.

    The gateway is a static proxy and mounts the routes either way, so the
    honest answer is that the endpoint exists with no model behind it.
    """
    if not settings.TTS_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="TTS is disabled (TTS_ENABLED=false)",
        )


def _validate(request: TtsRequest) -> None:
    """Reject what can be rejected before a model worker is occupied.

    The voice especially: checking it here means a typo costs nothing, while
    letting it reach the worker would tie up a GPU slot to produce a 422.
    """
    require_tts_enabled()
    if request.voice and request.voice not in VOICES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"unknown voice {request.voice!r}; choose one of {sorted(VOICES)}",
        )
    if request.stream and request.response_format != "pcm":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail='stream=true requires response_format="pcm"; a WAV header '
            "declares a total length that is unknown until the last clause",
        )


async def _iter_chunks(payload: dict) -> AsyncIterator[dict]:
    """Yield LitServe's NDJSON chunks, owning the connection for their life.

    The client and the response are entered here rather than by the caller
    because a StreamingResponse outlives the handler that returns it: closing
    either one at handler exit would truncate the audio mid-reply.
    """
    async with litserve_client.get_litserve_client() as client:
        async with litserve_client.synthesize_stream(client, payload) as response:
            if response.status_code >= 400:
                body = await response.aread()
                raise HTTPException(
                    status_code=response.status_code,
                    detail=_detail(body),
                )
            async for line in response.aiter_lines():
                if line.strip():
                    yield json.loads(line)


def _detail(body: bytes) -> str:
    try:
        return json.loads(body).get("detail", "TTS request failed")
    except (json.JSONDecodeError, AttributeError):
        return "TTS request failed"


def _pcm(chunk: dict) -> bytes:
    try:
        return base64.b64decode(chunk["audioContent"], validate=True)
    except (KeyError, binascii.Error, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LitServe returned a malformed audio payload",
        ) from exc


@router.post("/v1/audio/speech")
async def audio_speech(request: TtsRequest) -> Response:
    """OpenAI-compatible speech: text in, raw audio bytes out.

    The only TTS route, and the counterpart to POST /v1/audio/transcriptions,
    so a client written against that API reaches this service by base URL
    alone. "pcm" strips the WAV header, since a caller streaming into
    telephony wants frames rather than a container.

    The model worker always streams its clauses. With stream=false this
    reassembles them into one buffer, so that contract is unchanged; with
    stream=true they are forwarded as they arrive, which is what makes first
    audio arrive after the first clause instead of after the whole reply.
    """
    _validate(request)

    chunks = _iter_chunks(request.model_dump())
    try:
        first = await anext(chunks)
    except StopAsyncIteration as exc:
        await chunks.aclose()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LitServe produced no audio",
        ) from exc

    rate = int(first.get("sampleRate", 0))
    headers = {
        "X-Audio-Sample-Rate": str(rate),
        "X-Audio-Channels": "1",
    }

    if request.stream:

        async def frames() -> AsyncIterator[bytes]:
            yield _pcm(first)
            async for chunk in chunks:
                yield _pcm(chunk)

        return StreamingResponse(
            frames(),
            media_type="audio/pcm",
            headers=headers | {"X-Audio-Format": "pcm_s16le"},
        )

    pcm = bytearray(_pcm(first))
    async for chunk in chunks:
        pcm += _pcm(chunk)

    if request.response_format == "wav":
        return Response(
            content=wav_from_pcm(bytes(pcm), rate),
            media_type="audio/wav",
            headers=headers,
        )
    return Response(
        content=bytes(pcm),
        media_type="audio/pcm",
        headers=headers | {"X-Audio-Format": "pcm_s16le"},
    )
