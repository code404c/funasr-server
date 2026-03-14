"""funasr-server 应用工厂模块。

本模块负责创建和配置 FastAPI 应用实例，是整个服务的入口组装点。
主要职责包括：
- 初始化日志系统（loguru）
- 创建模型池和推理引擎
- 注册路由和依赖注入级鉴权
- 注册全局异常处理器
- 定义应用生命周期钩子（含可选模型预加载）

典型用法::

    from funasr_server.app import create_app
    app = create_app()  # 使用默认配置
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from funasr_server import __version__
from funasr_server.config import Settings, get_settings
from funasr_server.dependencies import ApiKeyDependency
from funasr_server.engine import FunASREngine
from funasr_server.errors import TranscriptionError
from funasr_server.model_pool import TTLModelPool
from funasr_server.routers.models import router as models_router
from funasr_server.routers.transcriptions import router as transcriptions_router


def _configure_logging(settings: Settings) -> None:
    """配置全局日志系统。

    使用 loguru 替代 Python 标准 logging，支持结构化 JSON 日志输出。
    每次调用会先移除所有已有的日志处理器，再按配置重新添加，
    确保日志行为与当前 settings 一致。

    Args:
        settings: 应用配置对象，使用其中的 ``log_level`` 和 ``log_json`` 两个字段。
    """
    logger.remove()
    logger.add(sys.stderr, level=settings.log_level, serialize=settings.log_json)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """FastAPI 应用生命周期管理器。

    启动时可选预加载指定模型（通过 ``FUNASR_PRELOAD_MODELS`` 配置），
    避免首次请求的冷启动延迟。

    Args:
        app: FastAPI 应用实例（由框架自动传入）。

    Yields:
        None: yield 之后应用开始接受请求，直到收到关闭信号。
    """
    logger.info("funasr-server starting up")

    # 模型预加载
    settings: Settings = app.state.settings
    engine: FunASREngine = app.state.engine
    for model_name in settings.preload_models:
        logger.info("Preloading model: {}", model_name)
        try:
            engine._get_or_load_model(model_name)
            logger.info("Model preloaded: {}", model_name)
        except Exception:
            logger.exception("Failed to preload model: {}", model_name)

    yield  # 应用在此处开始处理请求
    logger.info("funasr-server shutting down")


def create_app(*, settings: Settings | None = None) -> FastAPI:
    """应用工厂函数 —— 创建并返回完整配置的 FastAPI 实例。

    采用工厂模式（Factory Pattern），每次调用都会创建一个全新的 app 实例，
    这样做的好处是：
    1. 测试时可以传入自定义 settings，实现隔离测试
    2. 避免模块级别的全局状态，多进程部署更安全

    Args:
        settings: 可选的配置对象。如果不传，则通过 ``get_settings()`` 从环境变量读取。

    Returns:
        FastAPI: 配置完毕的应用实例，包含路由、中间件、引擎等全部组件。
    """
    settings = settings or get_settings()
    _configure_logging(settings)

    # 创建模型池和推理引擎
    model_pool: TTLModelPool[Any] = TTLModelPool(settings.model_ttl_seconds)
    engine = FunASREngine(settings=settings, model_pool=model_pool)

    app = FastAPI(title="funasr-server", version=__version__, lifespan=_lifespan)
    app.state.settings = settings
    app.state.engine = engine

    # ── 全局异常处理器：TranscriptionError → 结构化 JSON 响应 ──

    @app.exception_handler(TranscriptionError)
    async def transcription_error_handler(request: Request, exc: TranscriptionError):
        logger.warning("TranscriptionError: error_id={} message={}", exc.error_id, exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "message": exc.message,
                    "type": exc.error_type,
                    "error_id": exc.error_id,
                }
            },
        )

    # ── 可选 CORS 中间件 ──

    if settings.allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.allow_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # ── 注册路由（带 API Key 鉴权依赖） ──

    app.include_router(transcriptions_router, dependencies=[ApiKeyDependency])
    app.include_router(models_router, dependencies=[ApiKeyDependency])

    # /health 端点不需要鉴权，直接注册
    @app.get("/health")
    def health() -> dict[str, str]:
        """健康检查端点。"""
        return {"status": "ok"}

    return app
