import litserve as ls
from loguru import logger

from src.api.v1.asr.router import router as asr_router
from src.api.v1.live_cc.router import router as live_cc_router
from src.core.config import settings
from src.models.conformer.litapi import ASRLitAPI
from src.core.logging import configure_logging


def create_server() -> ls.LitServer:
    configure_logging()
    server = ls.LitServer(
        ASRLitAPI(max_batch_size=1, api_path=settings.API_PATH),
        accelerator=settings.ACCELERATOR,
        devices=settings.DEVICES,
        workers_per_device=settings.WORKERS_PER_DEVICE,
        healthcheck_path="/health",
    )
    server.app.include_router(asr_router)
    server.app.include_router(live_cc_router)
    return server


if __name__ == "__main__":
    server = create_server()
    logger.info(
        f"Starting Bangla ASR server ({settings.MODEL_NAME}) on {settings.HOST}:{settings.PORT}"
    )
    server.run(host=settings.HOST, port=settings.PORT, generate_client_file=False)
