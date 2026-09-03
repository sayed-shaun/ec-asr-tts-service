import sys

from loguru import logger

from src.core.config import settings

configured = False


def configure_logging() -> None:
    """Configure loguru sinks. No-op after the first call."""
    global configured
    if configured:
        return

    logger.remove()
    logger.add(sys.stderr, level=settings.LOG_LEVEL)
    logger.add(
        f"{settings.LOG_DIR}/{{time:YYYY-MM-DD}}.log",
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
            "- <level>{message}</level>"
        ),
        level=settings.LOG_LEVEL,
        rotation="00:00",
        retention="30 days",
        enqueue=True,
    )

    configured = True
