import litserve as ls

from src.core.config import settings
from src.litserver.litapi import ASRLitAPI


def create_litserve_server() -> ls.LitServer:
    """Build the LitServe server that holds the model, bound to LITSERVE_HOST:LITSERVE_PORT.

    Loopback-only by default — it's not the public entrypoint. The public
    gateway (src.api gateway app) is a pure FastAPI app with no model in it;
    it reaches this server over real HTTP, the same way any external client
    would.
    """
    return ls.LitServer(
        ASRLitAPI(max_batch_size=1, api_path=settings.API_PATH),
        accelerator=settings.ACCELERATOR,
        devices=settings.DEVICES,
        workers_per_device=settings.WORKERS_PER_DEVICE,
        healthcheck_path="/health",
    )
