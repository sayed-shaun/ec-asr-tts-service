"""Communication layer between the FastAPI gateway and the LitServe service.

Centralizes the base URL, the calls into LitServe (/predict for ASR,
/synthesize for TTS) and the error translation around them. Callers own the
client's lifetime (`async with get_litserve_client() as client: ...`) so a
long-lived connection can reuse one client across many calls.
"""

from typing import Awaitable

import httpx
from fastapi import HTTPException, status

from src.core.config import settings

LITSERVE_BASE_URL = settings.LITSERVE_BASE_URL
"""Where the model server lives. Defaults to the compose service name, so it
only resolves inside that network; deployments that split the two need it
set explicitly."""
PREDICT_PATH = "/predict"
SYNTHESIZE_PATH = "/synthesize"

DEFAULT_TIMEOUT = settings.LITSERVE_TIMEOUT + 10
"""Must stay above settings.LITSERVE_TIMEOUT so LitServe's own queue timeout
returns the 504, rather than this connection aborting first and racing it."""


def get_litserve_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=LITSERVE_BASE_URL)


async def forward_to_litserve(call: Awaitable[httpx.Response]) -> httpx.Response:
    """Await a litserve call, turning connection failures into distinguishable
    HTTP errors (502 unreachable, 504 no response in time) instead of a 500.
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


async def transcribe(
    client: httpx.AsyncClient, audio_content_b64: str, timeout: float = DEFAULT_TIMEOUT
) -> httpx.Response:
    """POST base64 audio to LitServe's /predict, return the raw response.

    Deliberately neither raises for status nor parses the body: callers want
    different things from it, so that policy stays theirs.
    """
    payload = {
        "config": {"language": {"sourceLanguage": "bn"}},
        "audio": [{"audioContent": audio_content_b64}],
    }
    return await client.post(PREDICT_PATH, json=payload, timeout=timeout)


async def transcribe_request(
    client: httpx.AsyncClient, payload: dict, timeout: float = DEFAULT_TIMEOUT
) -> httpx.Response:
    """POST an already-validated AsrRequest-shaped payload to /predict.

    For callers holding a full request body (multiple audio items, a
    non-default config) rather than one audio_content string.
    """
    return await client.post(PREDICT_PATH, json=payload, timeout=timeout)


async def synthesize(
    client: httpx.AsyncClient, payload: dict, timeout: float = DEFAULT_TIMEOUT
) -> httpx.Response:
    """POST an already-validated TtsRequest-shaped payload to /synthesize.

    That is the second LitAPI in the same model server, not a second service.
    Response is passed back unhandled, as in transcribe() above.
    """
    return await client.post(SYNTHESIZE_PATH, json=payload, timeout=timeout)
