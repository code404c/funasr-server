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

from functools import lru_cache  # 用于缓存 Settings 实例，避免重复解析环境变量
from pathlib import Path

from pydantic import SecretStr  # 敏感字段类型，打印/序列化时自动遮蔽明文
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用全局配置类。

    继承自 pydantic-settings 的 ``BaseSettings``，能自动从环境变量中读取值。
    字段名会被转换为大写并加上 ``FUNASR_`` 前缀后，与环境变量匹配。
    例如 ``log_level`` 对应环境变量 ``FUNASR_LOG_LEVEL``。

    Attributes:
        log_level: 日志级别，可选值为 DEBUG / INFO / WARNING / ERROR。
                   默认 ``"INFO"``。
        log_json: 是否以 JSON 格式输出日志。``True`` 时便于日志采集系统（如 ELK）解析，
                  本地开发可设为 ``False`` 获得更易读的输出。默认 ``True``。
        host: HTTP 服务监听地址。``"0.0.0.0"`` 表示监听所有网卡，
              容器内部署时通常保持默认。
        port: HTTP 服务监听端口，默认 ``8000``。
        model_cache_dir: FunASR / ModelScope 模型文件的缓存目录。
                         容器内默认 ``/modelscope-cache``，本地开发建议改为 ``~/models``。
        model_ttl_seconds: 模型池中模型的存活时间（秒）。超过此时间未被使用的模型会被卸载以释放
                           GPU 显存。设为 ``-1`` 表示永不过期。默认 ``900``（15 分钟）。
        device: PyTorch 推理设备标识符，如 ``"cuda:0"``、``"cpu"``。
                默认 ``"cuda:0"``，即使用第一块 GPU。
        api_key: 可选的 Bearer Token 鉴权密钥。类型为 ``SecretStr``，
                 在日志和序列化中不会泄露明文。为 ``None`` 时不启用鉴权。
    """

    # SettingsConfigDict 控制 pydantic-settings 的行为：
    #   env_prefix="FUNASR_"  → 环境变量前缀
    #   env_file=".env"       → 同时从项目根目录的 .env 文件加载
    #   extra="ignore"        → 遇到未知的环境变量时忽略，不报错
    model_config = SettingsConfigDict(env_prefix="FUNASR_", env_file=".env", extra="ignore")

    log_level: str = "INFO"
    log_json: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    model_cache_dir: Path = Path("/modelscope-cache")
    model_ttl_seconds: int = 900
    device: str = "cuda:0"
    api_key: SecretStr | None = None


@lru_cache
def get_settings() -> Settings:
    """获取全局唯一的配置实例（单例模式）。

    使用 ``@lru_cache`` 装饰器确保整个进程生命周期内只创建一次 ``Settings`` 对象。
    首次调用时会从环境变量和 ``.env`` 文件中解析配置，后续调用直接返回缓存结果。

    这种方式既避免了重复解析的开销，又保证了全局配置的一致性。
    在测试中，如果需要使用不同配置，可直接构造 ``Settings(...)`` 实例
    传给 ``create_app(settings=...)``，而不必调用此函数。

    Returns:
        Settings: 全局配置单例。
    """
    return Settings()
