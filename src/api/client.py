"""Communication layer between the FastAPI gateway and the LitServe service.

Centralizes the base URL and the /predict call the gateway makes into
LitServe so both asr/router.py and live_cc/router.py share one place
instead of each building its own litserve_base_url/httpx.AsyncClient.
Callers still own the client's lifetime
(`async with get_litserve_client() as client: ...`) so a long-lived
connection (e.g. a live-cc websocket) can reuse one client across many
calls, same as before this existed.
"""

import httpx

from src.core.config import settings

LITSERVE_BASE_URL = "http://litserver:8000"
PREDICT_PATH = "/predict"

# Must stay above settings.LITSERVE_TIMEOUT: LitServe's own queue timeout
# should be what returns a 504 to the client, not this connection aborting
# first and racing it.
DEFAULT_TIMEOUT = settings.LITSERVE_TIMEOUT + 10


def get_litserve_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=LITSERVE_BASE_URL)


async def transcribe(
    client: httpx.AsyncClient, audio_content_b64: str, timeout: float = DEFAULT_TIMEOUT
) -> httpx.Response:
    """POST base64 audio to LitServe's /predict, return the raw response.

    Deliberately does not call raise_for_status() or parse the body: callers
    want different things from it (asr/router.py passes both the JSON body
    *and* status code straight through to its own caller; live_cc/router.py
    just tries to pull output[].source and tolerates a malformed/error body
    by defaulting to empty text) — that policy stays theirs, not this
    module's.
    """
    payload = {
        "config": {"language": {"sourceLanguage": "bn"}},
        "audio": [{"audioContent": audio_content_b64}],
    }
    return await client.post(PREDICT_PATH, json=payload, timeout=timeout)


async def transcribe_request(
    client: httpx.AsyncClient, payload: dict, timeout: float = DEFAULT_TIMEOUT
) -> httpx.Response:
    """POST an already-built AsrRequest-shaped payload straight through to
    LitServe's /predict.

    For callers that already have a full request body validated (e.g. the
    raw JSON /transcribe route, which may carry multiple audio items or a
    non-default config) rather than a single audio_content string — see
    transcribe() above for that narrower case.
    """
    return await client.post(PREDICT_PATH, json=payload, timeout=timeout)
