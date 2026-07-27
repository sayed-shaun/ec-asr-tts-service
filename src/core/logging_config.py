import sys

from loguru import logger

from src.core.config import settings

_configured = False


def configure_logging() -> None:
    """Configure loguru sinks. Safe to call multiple times (no-op after the first)."""
    global _configured
    if _configured:
        return

    logger.remove()
    logger.add(sys.stderr, level=settings.log_level)
    logger.add(
        f"{settings.log_dir}/{{time:YYYY-MM-DD}}.log",
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        ),
        level=settings.log_level,
        rotation="00:00",
        retention="30 days",
        enqueue=True,
    )

    _configured = True
