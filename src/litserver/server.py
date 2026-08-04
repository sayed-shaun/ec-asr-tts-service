import litserve as ls

from src.core.config import settings
from src.litserver.litapi import ASRLitAPI


def create_litserve_server() -> ls.LitServer:
    """Build the LitServe server that holds the model.

    Not the public entrypoint — the gateway (src.api, pure FastAPI, no model
    in it) reaches this server over real HTTP, the same way any external
    client would.
    No api_path override: LitServe's own default is already "/predict".
    """
    return ls.LitServer(
        ASRLitAPI(max_batch_size=1),
        accelerator=settings.ACCELERATOR,
        devices=settings.DEVICES,
        workers_per_device=settings.WORKERS_PER_DEVICE,
        healthcheck_path="/health",
    )
