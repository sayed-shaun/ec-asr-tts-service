import base64
import json

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from src.api import client as litserve_client
from src.api.client import forward_to_litserve
from src.api.v1.asr.schema import AsrRequest, AsrResponse

router = APIRouter(tags=["ASR"])


async def transcribe_upload(file: UploadFile) -> JSONResponse:
    """Send one uploaded clip to LitServe and return its raw response.

    Only /v1/audio/transcriptions takes a file; POST /asr speaks the JSON
    contract instead. This process never loads the model: the upload is
    base64'd and forwarded to the model server over real HTTP.
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


@router.post("/asr", response_model=AsrResponse)
async def asr(request: AsrRequest) -> JSONResponse:
    """The Java service's contract: base64 clips in, transcripts out.

    No file upload -- a caller holding a file base64s it into `audio`, and the
    language rides nested under `config.language`. The body is forwarded to
    LitServe as-is rather than rebuilt, so `sourceLanguage` and any additional
    clips reach the model exactly as the client sent them. It sits at the root
    rather than under /api/v1 because Caddy routes /asr* straight here -- a
    path outside that prefix would need a proxy rule of its own.
    """
    async with litserve_client.get_litserve_client() as client:
        resp = await forward_to_litserve(
            litserve_client.transcribe_request(client, request.model_dump())
        )
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


@router.post("/v1/audio/transcriptions")
async def audio_transcriptions(file: UploadFile = File(...)) -> JSONResponse:
    """OpenAI-compatible transcription: multipart audio in, {"text": ...} out.

    The counterpart to POST /v1/audio/speech, so a client written against that
    API reaches this service by base URL alone. Segments are joined into one
    utterance because a caller feeding a chat turn wants the whole thing, not
    the service's internal split.
    """
    response = await transcribe_upload(file)
    if response.status_code >= 400:
        return response
    body = json.loads(bytes(response.body))
    text = " ".join(item.get("source", "") for item in body.get("output", []))
    return JSONResponse(content={"text": text.strip()})
