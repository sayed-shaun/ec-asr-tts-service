import litserve as ls
from loguru import logger

from src.api.v1.asr.lit_api import build_lit_api
from src.api.v1.asr.router import router as asr_router
from src.core.config import settings
from src.core.logging_config import configure_logging


def create_server() -> ls.LitServer:
    configure_logging()
    server = ls.LitServer(
        build_lit_api(),
        accelerator=settings.accelerator,
        devices=settings.devices,
        workers_per_device=settings.workers_per_device,
        healthcheck_path="/health",
    )
    server.app.include_router(asr_router)
    return server


if __name__ == "__main__":
    server = create_server()
    logger.info(
        f"Starting Bangla ASR server ({settings.model_name}) on {settings.host}:{settings.port}"
    )
    server.run(host=settings.host, port=settings.port, generate_client_file=False)
