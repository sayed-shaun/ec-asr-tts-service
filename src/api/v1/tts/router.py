import base64
import binascii
import io
import json
import wave

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse, Response

from src.api import client as litserve_client
from src.api.client import forward_to_litserve
from src.api.v1.tts.schema import TtsRequest
from src.core.config import settings
from src.litserver.parler.voices import VOICES

router = APIRouter(prefix="/api/v1/tts", tags=["TTS"])

openai_router = APIRouter(tags=["TTS"])
"""The OpenAI speech contract, mounted at the root rather than under
/api/v1/tts so a client written against that API needs only a base URL."""


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


@router.get("/info")
async def info() -> dict:
    """Static metadata about the deployed TTS model. Never touches it."""
    return {
        "model": settings.ACTIVE_TTS_MODEL_NAME,
        "enabled": settings.TTS_ENABLED,
        "language": "bn",
        "default_voice": settings.TTS_VOICE,
        "max_chars_per_chunk": settings.TTS_MAX_CHARS,
    }


@router.get("/voices")
async def voices() -> dict:
    """The voices /synthesize accepts, with the style prompt each maps to.

    Read locally rather than proxied: for Parler the voice list is a table of
    prompt strings, not model state.
    """
    return {"voices": VOICES, "default": settings.TTS_VOICE}


@router.post("/synthesize")
async def synthesize(request: TtsRequest) -> JSONResponse:
    """Text in, base64 WAV out, forwarded to LitServe's /synthesize.

    Prefer this over /synthesize/audio for any client that can read base64
    JSON; that route exists for Swagger UI and <audio> tags.
    """
    require_tts_enabled()
    if request.voice and request.voice not in VOICES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"unknown voice {request.voice!r}; choose one of {sorted(VOICES)}",
        )

    async with litserve_client.get_litserve_client() as client:
        resp = await forward_to_litserve(
            litserve_client.synthesize(client, request.model_dump())
        )

    return JSONResponse(content=resp.json(), status_code=resp.status_code)


@router.post("/synthesize/audio")
async def synthesize_audio(request: TtsRequest) -> Response:
    """Same synthesis, returned as a playable audio/wav body.

    Errors pass through as JSON so a failure stays readable; only the success
    path is binary.
    """
    resp = await synthesize(request)
    if resp.status_code >= 400:
        return resp

    body = json.loads(bytes(resp.body))
    try:
        wav = base64.b64decode(body["audioContent"], validate=True)
    except (KeyError, binascii.Error, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LitServe returned a malformed audio payload",
        ) from exc

    return Response(
        content=wav,
        media_type="audio/wav",
        headers={
            "X-Audio-Sample-Rate": str(body.get("sampleRate", "")),
            "X-Audio-Channels": "1",
        },
    )


@openai_router.post("/v1/audio/speech")
async def audio_speech(request: TtsRequest) -> Response:
    """OpenAI-compatible speech endpoint, returning raw audio bytes.

    Exists so a client already speaking that API can point at this service by
    base URL alone. It is a thin alias over /api/v1/tts/synthesize/audio;
    "pcm" strips the WAV header, since a caller streaming into telephony wants
    frames rather than a container.
    """
    response = await synthesize_audio(request)
    if response.status_code >= 400 or request.response_format == "wav":
        return response

    with wave.open(io.BytesIO(bytes(response.body)), "rb") as wav:
        frames = wav.readframes(wav.getnframes())
        rate = wav.getframerate()
    return Response(
        content=frames,
        media_type="audio/pcm",
        headers={
            "X-Audio-Format": "pcm_s16le",
            "X-Audio-Sample-Rate": str(rate),
            "X-Audio-Channels": "1",
        },
    )
