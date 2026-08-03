from fastapi import APIRouter

from src.api.model_info.schema import ModelInfoResponse
from src.core.config import settings
from src.litserver.engine import ASREngine

router = APIRouter(prefix="/internal/model", tags=["Model"])

num_parameters: int | None = None


@router.get("/info")
async def info() -> ModelInfoResponse:
    """Model name and parameter count, backing the gateway's
    /api/v1/asr/info route.

    Internal-only. Computed lazily on first call by loading the checkpoint
    on CPU just to count parameters (see ASREngine.count_parameters), then
    cached — the count never changes without a redeploy, and this process
    can't read it off an already-loaded GPU worker (see server.py).
    """
    global num_parameters
    if num_parameters is None:
        num_parameters = ASREngine.count_parameters(settings.MODEL_NAME)
    return ModelInfoResponse(
        model_name=settings.MODEL_NAME, num_parameters=num_parameters
    )
