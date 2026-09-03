import litserve as ls
from loguru import logger

from src.core.config import settings
from src.litserver.litapi import TTS_API_PATH, ASRLitAPI, TTSLitAPI


def create_litserve_server() -> ls.LitServer:
    """Build the LitServe server that holds the models.

    Not the public entrypoint: the gateway reaches it over real HTTP. Two
    LitAPIs share the process, ASR on "/predict" and, when TTS_ENABLED, TTS
    on "/synthesize". Each gets its own workers but they share
    accelerator/devices/workers_per_device, so both checkpoints sit on the
    same GPU. Budget for the pair, or set TTS_ENABLED=false.
    """
    apis: list[ls.LitAPI] = [ASRLitAPI(max_batch_size=1)]
    if settings.TTS_ENABLED:
        apis.append(TTSLitAPI(max_batch_size=1, api_path=TTS_API_PATH))
    else:
        logger.info("TTS_ENABLED=false — serving ASR only")

    return ls.LitServer(
        apis,
        accelerator=settings.ACCELERATOR,
        devices=settings.DEVICES,
        workers_per_device=settings.WORKERS_PER_DEVICE,
        healthcheck_path="/health",
        timeout=settings.LITSERVE_TIMEOUT,
    )
