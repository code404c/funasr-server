"""FastAPI 依赖注入模块。

本模块定义路由层使用的所有依赖项，替代从 request.app.state 直接取值的方式，
提供类型安全的参数注入，并让 Swagger UI 自动显示认证按钮。

依赖项清单：
- ``EngineDependency`` — 注入 FunASREngine 实例
- ``SettingsDependency`` — 注入 Settings 配置实例
- ``ApiKeyDependency`` — 路由级 Bearer Token 鉴权
- ``AudioFileDependency`` — 将上传文件落盘为临时文件，请求结束后自动清理
"""

from __future__ import annotations

import hmac
import tempfile
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from fastapi import Depends, HTTPException, Request, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from funasr_server.config import Settings
from funasr_server.engine import FunASREngine

# ── 引擎 & 配置依赖 ──


def get_engine(request: Request) -> FunASREngine:
    """从 app.state 获取 FunASR 推理引擎实例。"""
    return request.app.state.engine


def get_app_settings(request: Request) -> Settings:
    """从 app.state 获取应用配置实例。"""
    return request.app.state.settings


EngineDependency = Annotated[FunASREngine, Depends(get_engine)]
SettingsDependency = Annotated[Settings, Depends(get_app_settings)]


# ── API Key 鉴权依赖 ──

# auto_error=False: 缺少 Authorization 头时不自动抛 403，由我们返回 401
_bearer_scheme = HTTPBearer(auto_error=False)


async def verify_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> None:
    """验证 Bearer Token 的依赖函数。

    - 如果 settings.api_key 未配置，直接放行（无鉴权模式）
    - 使用 hmac.compare_digest 做恒定时间比较，防止时序攻击
    """
    settings: Settings = request.app.state.settings
    if not settings.api_key:
        return

    expected = settings.api_key.get_secret_value()
    token = credentials.credentials if credentials else ""

    if not hmac.compare_digest(token, expected):
        from loguru import logger

        client = request.client.host if request.client else "unknown"
        logger.warning("API key auth failed: path={} client={}", request.url.path, client)
        raise HTTPException(status_code=401, detail="Invalid API key")


ApiKeyDependency = Depends(verify_api_key)


# ── 音频文件依赖 ──


@dataclass
class AudioFile:
    """音频临时文件封装，自动管理生命周期。"""

    path: Path
    filename: str

    def cleanup(self) -> None:
        """删除临时文件。"""
        self.path.unlink(missing_ok=True)


async def get_audio_file(file: UploadFile) -> AsyncGenerator[AudioFile, None]:
    """将上传的音频文件写入临时目录并返回 AudioFile。

    使用 yield 依赖模式确保请求结束后自动清理临时文件。
    """
    suffix = Path(file.filename).suffix if file.filename else ".wav"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        content = await file.read()
        tmp_path.write_bytes(content)
        yield AudioFile(path=tmp_path, filename=file.filename or "audio.wav")
    finally:
        tmp_path.unlink(missing_ok=True)


AudioFileDependency = Annotated[AudioFile, Depends(get_audio_file)]
