from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.core.config import settings
from src.litserver.speaker import SpeakerEncoder
from src.utils.audio import decode_base64_audio

router = APIRouter(prefix="/internal/speaker", tags=["speaker"])

_encoder = SpeakerEncoder()


class EmbedRequest(BaseModel):
    audio_content: str


@router.post("/embed")
async def embed(payload: EmbedRequest) -> dict:
    """Compute a speaker embedding for one audio chunk.

    Internal-only: called by the gateway's live-cc speaker gate (see
    src/api/v1/live_cc/router.py), never by external clients. Runs in
    LitServe's main process, not a GPU worker — Resemblyzer's model is small
    enough that a dedicated worker pool isn't warranted.
    """
    try:
        waveform = decode_base64_audio(
            payload.audio_content, target_sr=settings.SAMPLE_RATE
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    embedding = _encoder.embed(waveform, sample_rate=settings.SAMPLE_RATE)
    return {"embedding": embedding}
