import base64
from typing import Awaitable

import httpx
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from src.api import client as litserve_client
from src.api.v1.asr.schema import AsrRequest
from src.core.config import settings

router = APIRouter(prefix="/api/v1/asr", tags=["ASR"])


async def forward_to_litserve(call: Awaitable[httpx.Response]) -> httpx.Response:
    """Await a litserve_client call, turning connection failures into HTTP
    errors a client can distinguish (502: litserver unreachable, 504: it
    accepted the connection but didn't respond in time) instead of a bare 500.
    """
    try:
        return await call
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="LitServe request timed out",
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail="LitServe is unreachable"
        ) from exc


@router.get("/info")
async def info() -> JSONResponse:
    """Model metadata: name and parameter count (from LitServe's internal
    /internal/model/info), plus language and sample rate (static gateway
    config) — see README's Endpoints table.
    """
    async with litserve_client.get_litserve_client() as client:
        resp = await forward_to_litserve(litserve_client.model_info(client))

    if resp.status_code != status.HTTP_200_OK:
        return JSONResponse(content=resp.json(), status_code=resp.status_code)

    model_meta = resp.json()
    return JSONResponse(
        content={
            "model_name": model_meta["model_name"],
            "num_parameters_millions": round(model_meta["num_parameters"] / 1e6, 1),
            "language": "bn",
            "sample_rate": settings.SAMPLE_RATE,
        }
    )


@router.post("/transcribe")
async def transcribe(request: AsrRequest) -> JSONResponse:
    """Raw JSON inference — same request/response contract as LitServe's
    internal /predict, forwarded as a real HTTP request (this process never
    loads the model itself). Prefer this over /transcribe/file for any
    client that can send base64 JSON directly; /transcribe/file exists only
    because Swagger UI can't render a base64 text box nicely.
    """
    async with litserve_client.get_litserve_client() as client:
        resp = await forward_to_litserve(
            litserve_client.transcribe_request(client, request.model_dump())
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
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty"
        )

    audio_content_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    async with litserve_client.get_litserve_client() as client:
        resp = await forward_to_litserve(
            litserve_client.transcribe(client, audio_content_b64)
        )

    return JSONResponse(content=resp.json(), status_code=resp.status_code)
