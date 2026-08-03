import litserve as ls

from src.api.model_info.router import router as model_info_router
from src.api.speaker.router import router as speaker_router
from src.core.config import settings
from src.litserver.litapi import ASRLitAPI


def create_litserve_server() -> ls.LitServer:
    """Build the LitServe server that holds the model.

    Not the public entrypoint — the gateway (src.api, pure FastAPI, no model
    in it) reaches this server over real HTTP, the same way any external
    client would. Also mounts two internal routes that run in this main
    process rather than a GPU worker: speaker-embedding (used by the
    gateway's live-cc speaker gate, SPEAKER_GATE_ENABLED — Resemblyzer's model
    is small enough not to need a worker) and model-info (parameter count for
    the gateway's /api/v1/asr/info, computed by loading the checkpoint
    separately since a worker's already-loaded model isn't reachable here).
    No api_path override: LitServe's own default is already "/predict".
    """
    server = ls.LitServer(
        ASRLitAPI(max_batch_size=1),
        accelerator=settings.ACCELERATOR,
        devices=settings.DEVICES,
        workers_per_device=settings.WORKERS_PER_DEVICE,
        healthcheck_path="/health",
    )
    server.app.include_router(speaker_router)
    server.app.include_router(model_info_router)
    return server
