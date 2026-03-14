"""funasr-server 配置模块。

本模块定义了服务的全部可配置项，基于 pydantic-settings 实现，
支持从环境变量和 ``.env`` 文件中自动加载配置值。

所有环境变量均以 ``FUNASR_`` 为前缀，例如：
- ``FUNASR_DEVICE=cuda:0`` 对应 ``Settings.device``
- ``FUNASR_MODEL_TTL_SECONDS=900`` 对应 ``Settings.model_ttl_seconds``

典型用法::

    from funasr_server.config import get_settings
    settings = get_settings()
    print(settings.device)  # "cuda:0"
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用全局配置类。

    继承自 pydantic-settings 的 ``BaseSettings``，能自动从环境变量中读取值。
    字段名会被转换为大写并加上 ``FUNASR_`` 前缀后，与环境变量匹配。

    Attributes:
        log_level: 日志级别，默认 ``"INFO"``。
        log_json: 是否以 JSON 格式输出日志，默认 ``True``。
        host: HTTP 服务监听地址，默认 ``"0.0.0.0"``。
        port: HTTP 服务监听端口，默认 ``8000``。
        model_cache_dir: 模型缓存目录，默认 ``/modelscope-cache``。
        model_ttl_seconds: 模型 TTL（秒），``-1`` 永不过期，默认 ``900``。
        device: PyTorch 推理设备，默认 ``"cuda:0"``。
        batch_size_s: VAD 切分后每批最大秒数，默认 ``60``。
        merge_length_s: VAD 合并后每段最大秒数，默认 ``15``。
        api_key: 可选的 Bearer Token 鉴权密钥。
        preload_models: 启动时预加载的模型列表，默认为空。
        allow_origins: CORS 允许的源列表，为 None 时不启用 CORS。
    """

    model_config = SettingsConfigDict(env_prefix="FUNASR_", env_file=".env", extra="ignore")

    log_level: str = "INFO"
    log_json: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    model_cache_dir: Path = Path("/modelscope-cache")
    model_ttl_seconds: int = 900
    device: str = "cuda:0"
    batch_size_s: int = 60
    merge_length_s: int = 15
    api_key: SecretStr | None = None
    preload_models: list[str] = []
    allow_origins: list[str] | None = None


@lru_cache
def get_settings() -> Settings:
    """获取全局唯一的配置实例（单例模式）。

    使用 ``@lru_cache`` 装饰器确保整个进程生命周期内只创建一次 ``Settings`` 对象。

    Returns:
        Settings: 全局配置单例。
    """
    return Settings()
