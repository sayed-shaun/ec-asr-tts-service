import base64

import httpx
from fastapi import APIRouter, File, HTTPException, UploadFile
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
        "language": "bn",
        "sample_rate": settings.SAMPLE_RATE,
        "predict_endpoint": settings.API_PATH,
    }


@router.post("/transcribe/file")
async def transcribe_file(file: UploadFile = File(...)) -> JSONResponse:
    """Swagger-friendly file-upload wrapper around POST {api_path}.

    The real inference endpoint takes base64-encoded JSON (so any client, not
    just browsers, can drive it) which Swagger UI can only render as a plain
    text box. This gives Swagger UI's "Choose File" button something to call,
    forwarding as a real HTTP request to the LitServe model server (this
    process never loads the model itself).
    """
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    payload = {
        "config": {"language": {"sourceLanguage": "bn"}},
        "audio": [{"audioContent": base64.b64encode(audio_bytes).decode("utf-8")}],
    }

    litserve_base_url = f"http://{settings.LITSERVE_HOST}:{settings.LITSERVE_PORT}"
    async with httpx.AsyncClient(base_url=litserve_base_url) as client:
        resp = await client.post(settings.API_PATH, json=payload, timeout=120)

    return JSONResponse(content=resp.json(), status_code=resp.status_code)
