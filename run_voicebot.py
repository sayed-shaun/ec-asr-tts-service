import uvicorn
from loguru import logger

from src.core.config import settings
from src.core.logging import configure_logging
from src.voicebot.app import create_voicebot_app

if __name__ == "__main__":
    configure_logging()
    app = create_voicebot_app()
    logger.info(
        f"Starting voicebot streaming service on 0.0.0.0:{settings.VOICEBOT_PORT}"
    )
    uvicorn.run(app, host="0.0.0.0", port=settings.VOICEBOT_PORT)
