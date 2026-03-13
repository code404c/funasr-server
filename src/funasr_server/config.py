from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
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
    return Settings()
