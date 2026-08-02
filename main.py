import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from loguru import logger

from src.api.v1.asr.router import router as asr_router
from src.api.v1.live_cc.router import router as live_cc_router
from src.core.config import settings
from src.core.logging import configure_logging


def create_gateway_app() -> FastAPI:
    """Build the public-facing pure FastAPI app: ASR + live-cc routers and the
    static test GUI.

    No model is loaded in this process — it's a thin proxy in front of the
    LitServe model server (see run_litserve.py), reached over real HTTP at
    litserver:8000 (fixed, see src.core.config). /static serves the manual
    test GUI for /predict and /v1/live-cc/ws (static/index.html).
    """
    app = FastAPI()

    @app.get("/api/v1/asr/info")
    async def info() -> dict:
        """Static metadata about the deployed model/endpoint. Safe to call
        from the main process — unlike /predict, it does not touch the
        model itself.
        """
        return {
            "model": settings.MODEL_NAME,
            "language": "bn",
            "sample_rate": settings.SAMPLE_RATE,
        }

    app.include_router(asr_router)
    app.include_router(live_cc_router)
    app.mount("/static", StaticFiles(directory="static"), name="static")
    return app


if __name__ == "__main__":
    configure_logging()
    gateway_app = create_gateway_app()
    logger.info(f"Starting Bangla ASR gateway on 0.0.0.0:{settings.GATEWAY_PORT}")
    uvicorn.run(gateway_app, host="0.0.0.0", port=settings.GATEWAY_PORT)
