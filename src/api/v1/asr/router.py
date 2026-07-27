import base64

import httpx
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from src.core.config import settings

router = APIRouter(prefix="/v1/asr", tags=["asr"])


@router.get("/info")
async def info() -> dict:
    """Static metadata about the deployed model/endpoint. Safe to call from the
    main process — unlike /predict, it does not touch the model itself.
    """
    return {
        "model": settings.MODEL_NAME,
        "architecture": "FastConformer-CTC (NeMo)",
        "language": "bn",
        "sample_rate": settings.SAMPLE_RATE,
        "predict_endpoint": settings.API_PATH,
    }


@router.post("/transcribe/file")
async def transcribe_file(request: Request, file: UploadFile = File(...)) -> JSONResponse:
    """Swagger-friendly file-upload wrapper around POST {api_path}.

    The real inference endpoint takes base64-encoded JSON (so any client, not
    just browsers, can drive it) which Swagger UI can only render as a plain
    text box. This gives Swagger UI's "Choose File" button something to call,
    and re-dispatches into the same app so it still goes through the actual
    LitServe worker/model — nothing is duplicated here.
    """
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    payload = {
        "config": {"language": {"sourceLanguage": "bn"}},
        "audio": [{"audioContent": base64.b64encode(audio_bytes).decode("utf-8")}],
    }

    transport = httpx.ASGITransport(app=request.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://internal") as client:
        resp = await client.post(settings.API_PATH, json=payload, timeout=120)

    return JSONResponse(content=resp.json(), status_code=resp.status_code)
