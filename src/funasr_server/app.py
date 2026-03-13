from __future__ import annotations

import hmac
import sys
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger

from funasr_server.config import Settings, get_settings
from funasr_server.engine import FunASREngine
from funasr_server.model_pool import TTLModelPool
from funasr_server.routers.transcriptions import router as transcriptions_router


def _configure_logging(settings: Settings) -> None:
    logger.remove()
    logger.add(sys.stderr, level=settings.log_level, serialize=settings.log_json)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    logger.info("funasr-server starting up")
    yield
    logger.info("funasr-server shutting down")


def create_app(*, settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    _configure_logging(settings)

    model_pool: TTLModelPool[Any] = TTLModelPool(settings.model_ttl_seconds)
    engine = FunASREngine(settings=settings, model_pool=model_pool)

    app = FastAPI(title="funasr-server", version="0.1.0", lifespan=_lifespan)
    app.state.settings = settings
    app.state.engine = engine

    if settings.api_key:
        expected = settings.api_key.get_secret_value()

        @app.middleware("http")
        async def verify_api_key(request: Request, call_next):
            if request.url.path == "/health":
                return await call_next(request)
            auth = request.headers.get("Authorization", "")
            token = auth.removeprefix("Bearer ").strip()
            if not hmac.compare_digest(token, expected):
                client = request.client.host if request.client else "unknown"
                logger.warning("API key auth failed: path={} client={}", request.url.path, client)
                return JSONResponse(status_code=401, content={"detail": "Invalid API key"})
            return await call_next(request)

    app.include_router(transcriptions_router)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app
