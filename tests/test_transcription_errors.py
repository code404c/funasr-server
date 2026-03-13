"""transcriptions.py error handling 分支测试 — 覆盖 FunASRUnavailableError, ValueError, generic Exception。"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from funasr_server.app import create_app
from funasr_server.config import Settings
from funasr_server.engine import FunASRUnavailableError


@pytest.fixture
def settings() -> Settings:
    return Settings(model_cache_dir=Path("/tmp/test-cache"), device="cpu")


def _upload(client: TestClient, **kwargs):
    defaults = {"model": "cn_meeting", "response_format": "json"}
    defaults.update(kwargs)
    files = {"file": ("test.wav", BytesIO(b"fake-audio"), "audio/wav")}
    return client.post("/v1/audio/transcriptions", files=files, data=defaults)


class TestTranscriptionErrorHandling:
    """转写端点的异常处理分支测试。"""

    def test_funasr_unavailable_returns_503(self, settings: Settings) -> None:
        """FunASRUnavailableError 应返回 503。"""
        app = create_app(settings=settings)
        with patch.object(
            app.state.engine,
            "transcribe",
            side_effect=FunASRUnavailableError("FunASR is not installed"),
        ):
            client = TestClient(app)
            resp = _upload(client)

        assert resp.status_code == 503
        assert "FunASR is not installed" in resp.json()["detail"]

    def test_value_error_returns_400(self, settings: Settings) -> None:
        """ValueError 应返回 400。"""
        app = create_app(settings=settings)
        with patch.object(
            app.state.engine,
            "transcribe",
            side_effect=ValueError("Invalid model: bad_model"),
        ):
            client = TestClient(app)
            resp = _upload(client)

        assert resp.status_code == 400
        assert "Invalid model" in resp.json()["detail"]

    def test_generic_exception_returns_500(self, settings: Settings) -> None:
        """其他未知异常应返回 500。"""
        app = create_app(settings=settings)
        with patch.object(
            app.state.engine,
            "transcribe",
            side_effect=RuntimeError("Something went wrong"),
        ):
            client = TestClient(app)
            resp = _upload(client)

        assert resp.status_code == 500
        assert resp.json()["detail"] == "Internal transcription error"
