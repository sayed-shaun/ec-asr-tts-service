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

router = APIRouter(tags=["TTS"])


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


async def synthesize_request(request: TtsRequest) -> JSONResponse:
    """Send one synthesis request to LitServe and return its raw response.

    The voice is checked here so an unknown one fails before occupying a
    model worker.
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


@router.post("/v1/audio/speech")
async def audio_speech(request: TtsRequest) -> Response:
    """OpenAI-compatible speech: text in, raw audio bytes out.

    The only TTS route, and the counterpart to POST /v1/audio/transcriptions,
    so a client written against that API reaches this service by base URL
    alone. "pcm" strips the WAV header, since a caller streaming into
    telephony wants frames rather than a container.
    """
    response = await synthesize_request(request)
    if response.status_code >= 400:
        return response

    body = json.loads(bytes(response.body))
    try:
        wav = base64.b64decode(body["audioContent"], validate=True)
    except (KeyError, binascii.Error, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LitServe returned a malformed audio payload",
        ) from exc

    if request.response_format == "wav":
        return Response(
            content=wav,
            media_type="audio/wav",
            headers={
                "X-Audio-Sample-Rate": str(body.get("sampleRate", "")),
                "X-Audio-Channels": "1",
            },
        )

    with wave.open(io.BytesIO(wav), "rb") as container:
        frames = container.readframes(container.getnframes())
        rate = container.getframerate()
    return Response(
        content=frames,
        media_type="audio/pcm",
        headers={
            "X-Audio-Format": "pcm_s16le",
            "X-Audio-Sample-Rate": str(rate),
            "X-Audio-Channels": "1",
        },
    )
