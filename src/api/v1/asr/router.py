import base64
import json

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from loguru import logger

from src.api import client as litserve_client
from src.api.client import forward_to_litserve
from src.api.v1.asr.schema import AsrRequest
from src.core.config import settings

router = APIRouter(prefix="/api/v1/asr", tags=["ASR"])

compat_router = APIRouter(tags=["ASR"])
"""Paths the EC chatbot UI calls directly through Caddy, which reverse-proxies
/asr* to this service rather than through the chatbot backend."""

openai_router = APIRouter(tags=["ASR"])
"""The OpenAI transcription contract, mounted at the root so a client written
against that API reaches this service by base URL alone."""


@router.get("/info")
async def info() -> dict:
    """Static metadata about the deployed model. Never touches the model."""
    return {
        "model": settings.ACTIVE_MODEL_NAME,
        "language": "bn",
        "sample_rate": settings.SAMPLE_RATE,
    }


@router.post("/transcribe")
async def transcribe(request: AsrRequest) -> JSONResponse:
    """Raw JSON inference, forwarded to LitServe's /predict over real HTTP.

    Prefer this over /transcribe/file for any client that can send base64
    JSON directly.
    """
    async with litserve_client.get_litserve_client() as client:
        resp = await forward_to_litserve(
            litserve_client.transcribe_request(client, request.model_dump())
        )

    return JSONResponse(content=resp.json(), status_code=resp.status_code)


@router.post("/transcribe/file")
async def transcribe_file(file: UploadFile = File(...)) -> JSONResponse:
    """Multipart-upload wrapper around /transcribe.

    Exists because Swagger UI can only render the real base64-JSON endpoint
    as a plain text box.
    """
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty"
        )

    audio_content_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    async with litserve_client.get_litserve_client() as client:
        resp = await forward_to_litserve(
            litserve_client.transcribe(client, audio_content_b64)
        )

    return JSONResponse(content=resp.json(), status_code=resp.status_code)


@openai_router.post("/v1/audio/transcriptions")
async def audio_transcriptions(file: UploadFile = File(...)) -> JSONResponse:
    """OpenAI-compatible transcription: multipart audio in, {"text": ...} out.

    The counterpart to POST /v1/audio/speech. Segments are joined into one
    utterance because a caller feeding a chat turn wants the whole thing, not
    the service's internal split.
    """
    response = await transcribe_file(file)
    if response.status_code >= 400:
        return response
    body = json.loads(bytes(response.body))
    text = " ".join(item.get("source", "") for item in body.get("output", []))
    return JSONResponse(content={"text": text.strip()})


@compat_router.post("/asr")
async def asr_upload(
    file: UploadFile = File(...), model_type: str = Form(default="")
) -> JSONResponse:
    """Multipart upload returning this pipeline's own {output: [{source}]}.

    What the chatbot UI posts recorded audio to. It sits at the root, not
    under /api/v1, because Caddy routes /asr* straight here -- a path outside
    that prefix would need a proxy rule of its own.

    model_type is accepted and ignored: the caller names an engine, and this
    service serves exactly one.
    """
    if model_type and model_type != "zipformer":
        logger.info(f"ignoring requested model_type '{model_type}'")
    return await transcribe_file(file)
