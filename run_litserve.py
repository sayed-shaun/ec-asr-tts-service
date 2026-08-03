from loguru import logger

from src.core.config import settings
from src.core.logging import configure_logging
from src.litserver.server import create_litserve_server

if __name__ == "__main__":
    configure_logging()
    server = create_litserve_server()
    logger.info(
        f"Starting LitServe model server ({settings.MODEL_NAME}) on " f"0.0.0.0:8000"
    )
    server.run(host="0.0.0.0", port=8000, generate_client_file=False)
