from loguru import logger

from src.core.config import settings
from src.core.logging import configure_logging
from src.litserver.server import create_litserve_server

if __name__ == "__main__":
    configure_logging()
    server = create_litserve_server()
    models = settings.ACTIVE_MODEL_NAME
    if settings.ACTIVE_TTS_MODEL_NAME:
        models += f" + {settings.ACTIVE_TTS_MODEL_NAME}"
    logger.info(
        f"Starting LitServe model server ({models}) on "
        f"0.0.0.0:{settings.LITSERVE_PORT}"
    )
    server.run(
        host="0.0.0.0", port=settings.LITSERVE_PORT, generate_client_file=False
    )
