import base64

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from src.api import client as litserve_client
from src.api.v1.asr.schema import AsrRequest

router = APIRouter(prefix="/api/v1/asr", tags=["asr"])


@router.post("/transcribe")
async def transcribe(request: AsrRequest) -> JSONResponse:
    """Raw JSON inference — same request/response contract as LitServe's
    internal /predict, forwarded as a real HTTP request (this process never
    loads the model itself). Prefer this over /transcribe/file for any
    client that can send base64 JSON directly; /transcribe/file exists only
    because Swagger UI can't render a base64 text box nicely.
    """
    async with litserve_client.get_litserve_client() as client:
        resp = await litserve_client.transcribe_request(
            client, request.model_dump()
        )

    return JSONResponse(content=resp.json(), status_code=resp.status_code)


@router.post("/transcribe/file")
async def transcribe_file(file: UploadFile = File(...)) -> JSONResponse:
    """Swagger-friendly file-upload wrapper around POST {predict_path}.

    The real inference endpoint takes base64-encoded JSON (so any client, not
    just browsers, can drive it) which Swagger UI can only render as a plain
    text box. This gives Swagger UI's "Choose File" button something to call,
    forwarding as a real HTTP request to the LitServe model server (this
    process never loads the model itself).
    """
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    audio_content_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    async with litserve_client.get_litserve_client() as client:
        resp = await litserve_client.transcribe(client, audio_content_b64)

    return JSONResponse(content=resp.json(), status_code=resp.status_code)
