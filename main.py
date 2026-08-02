import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from loguru import logger

from src.api.v1.asr.router import router as asr_router
from src.api.v1.live_cc.router import router as live_cc_router
from src.core.config import settings
from src.core.logging import configure_logging


def create_gateway_app() -> FastAPI:
    """Build the public-facing pure FastAPI app: ASR + live-cc routers and the static test GUI.

    No model is loaded in this process — it's a thin proxy in front of the
    LitServe model server (see run_litserve.py), reached over real HTTP at
    ASR_LITSERVE_HOST:ASR_LITSERVE_PORT. /static serves the manual test GUI
    for /predict and /v1/live-cc/ws (static/index.html).
    """
    app = FastAPI()
    app.include_router(asr_router)
    app.include_router(live_cc_router)
    app.mount("/static", StaticFiles(directory="static"), name="static")
    return app


if __name__ == "__main__":
    configure_logging()
    gateway_app = create_gateway_app()
    logger.info(f"Starting Bangla ASR gateway on {settings.HOST}:{settings.API_PORT}")
    uvicorn.run(gateway_app, host=settings.HOST, port=settings.API_PORT)
