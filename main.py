import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.v1.asr.router import router as asr_router
from src.api.v1.tts.router import router as tts_router
from src.core.config import settings
from src.core.logging import configure_logging


def create_gateway_app() -> FastAPI:
    """Build the public-facing FastAPI app: the ASR and TTS routers.

    Loads no model: a thin proxy in front of the LitServe model server at
    LITSERVE_BASE_URL.
    """
    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in settings.CORS_ALLOW_ORIGINS.split(",")],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> dict:
        """Liveness for the gateway itself. Loads no model, so it answers even
        while the model server is still warming up."""
        return {"status": "ok", "asr": settings.ACTIVE_MODEL_NAME,
                "tts": settings.ACTIVE_TTS_MODEL_NAME}

    app.include_router(asr_router)
    app.include_router(tts_router)
    return app


if __name__ == "__main__":
    configure_logging()
    gateway_app = create_gateway_app()
    logger.info(f"Starting Bangla ASR gateway on 0.0.0.0:{settings.GATEWAY_PORT}")
    uvicorn.run(gateway_app, host="0.0.0.0", port=settings.GATEWAY_PORT)
