"""funasr-server 应用工厂模块。

本模块负责创建和配置 FastAPI 应用实例，是整个服务的入口组装点。
主要职责包括：
- 初始化日志系统（loguru）
- 创建模型池和推理引擎
- 注册路由和中间件（含可选的 API Key 鉴权）
- 定义应用生命周期钩子

典型用法::

    from funasr_server.app import create_app
    app = create_app()  # 使用默认配置
"""

from __future__ import annotations

import hmac  # 用于安全的字符串比较，防止时序攻击
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
    """配置全局日志系统。

    使用 loguru 替代 Python 标准 logging，支持结构化 JSON 日志输出。
    每次调用会先移除所有已有的日志处理器，再按配置重新添加，
    确保日志行为与当前 settings 一致。

    Args:
        settings: 应用配置对象，使用其中的 ``log_level``（日志级别，如 INFO/DEBUG）
                  和 ``log_json``（是否以 JSON 格式输出日志）两个字段。
    """
    # 移除 loguru 默认的 stderr handler，避免重复输出
    logger.remove()
    # serialize=True 时 loguru 会把每条日志序列化为 JSON，便于日志采集系统解析
    logger.add(sys.stderr, level=settings.log_level, serialize=settings.log_json)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """FastAPI 应用生命周期管理器。

    利用 Python 异步上下文管理器，在 ``yield`` 之前执行启动逻辑，
    ``yield`` 之后执行关闭逻辑。FastAPI 会在服务启动和关闭时自动调用。

    Args:
        app: FastAPI 应用实例（由框架自动传入）。

    Yields:
        None: yield 之后应用开始接受请求，直到收到关闭信号。
    """
    logger.info("funasr-server starting up")
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
                  关键字参数（keyword-only），调用时必须写 ``create_app(settings=s)``。

    Returns:
        FastAPI: 配置完毕的应用实例，包含路由、中间件、引擎等全部组件。
    """
    # 如果未传入 settings，则从环境变量 / .env 文件中加载默认配置
    settings = settings or get_settings()
    _configure_logging(settings)

    # 创建模型池：TTLModelPool 负责缓存已加载的 ASR 模型，超过 TTL 自动卸载以释放 GPU 显存
    model_pool: TTLModelPool[Any] = TTLModelPool(settings.model_ttl_seconds)
    # 创建推理引擎：封装了 FunASR 的模型加载和推理逻辑
    engine = FunASREngine(settings=settings, model_pool=model_pool)

    # 创建 FastAPI 实例，lifespan 参数指定生命周期管理器
    app = FastAPI(title="funasr-server", version="0.1.0", lifespan=_lifespan)
    # 将配置和引擎挂载到 app.state，路由处理函数中可通过 request.app.state 访问
    app.state.settings = settings
    app.state.engine = engine

    # ── 可选的 API Key 鉴权中间件 ──
    # 仅当环境变量 FUNASR_API_KEY 非空时才启用鉴权
    if settings.api_key:
        # 提前取出明文密钥，避免每次请求都调用 get_secret_value()
        expected = settings.api_key.get_secret_value()

        @app.middleware("http")
        async def verify_api_key(request: Request, call_next):
            """HTTP 中间件：校验请求头中的 Bearer Token。

            对每个请求（除 /health 外）检查 Authorization 头，
            使用 hmac.compare_digest 做恒定时间比较，防止时序攻击。

            Args:
                request: 当前 HTTP 请求对象。
                call_next: 调用链中的下一个处理器（中间件或路由）。

            Returns:
                Response: 鉴权通过则返回下游响应，否则返回 401 JSON 响应。
            """
            # /health 端点不需要鉴权，用于负载均衡器的健康检查
            if request.url.path == "/health":
                return await call_next(request)
            # 从 Authorization 头中提取 token，格式为 "Bearer <token>"
            auth = request.headers.get("Authorization", "")
            token = auth.removeprefix("Bearer ").strip()
            # hmac.compare_digest 是恒定时间比较，即使 token 不匹配也不会提前返回，
            # 防止攻击者通过测量响应时间逐字符猜测密钥（时序攻击）
            if not hmac.compare_digest(token, expected):
                client = request.client.host if request.client else "unknown"
                logger.warning("API key auth failed: path={} client={}", request.url.path, client)
                return JSONResponse(status_code=401, content={"detail": "Invalid API key"})
            return await call_next(request)

    # 注册转写路由（/v1/audio/transcriptions）
    app.include_router(transcriptions_router)

    @app.get("/health")
    def health() -> dict[str, str]:
        """健康检查端点。

        供 Docker 健康检查、Kubernetes liveness probe 或负载均衡器调用，
        返回固定的 ``{"status": "ok"}`` 表示服务正常运行。

        Returns:
            dict: 包含 ``status`` 字段的字典。
        """
        return {"status": "ok"}

    return app
