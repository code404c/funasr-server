"""app.py 认证与 lifespan 测试 — 覆盖 API key 鉴权依赖和启动/关闭事件。"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from funasr_server.app import create_app
from funasr_server.config import Settings
from funasr_server.schemas import Segment, TranscriptionResult


@pytest.fixture
def fake_result() -> TranscriptionResult:
    return TranscriptionResult(
        text="hello",
        language="zh",
        duration=1.0,
        segments=[Segment(id=0, start=0.0, end=1.0, text="hello", x_speaker_id=None)],
        x_speakers=[],
    )


def _make_client(*, api_key: str | None = None, fake_result: TranscriptionResult) -> TestClient:
    """创建带可选 api_key 的测试客户端。"""
    settings = Settings(model_cache_dir=Path("/tmp/test-cache"), device="cpu", api_key=api_key)
    app = create_app(settings=settings)
    with patch.object(app.state.engine, "transcribe", return_value=fake_result):
        yield TestClient(app)


@pytest.fixture
def client_with_key(fake_result: TranscriptionResult):
    yield from _make_client(api_key="test-secret-key", fake_result=fake_result)


@pytest.fixture
def client_no_key(fake_result: TranscriptionResult):
    yield from _make_client(api_key=None, fake_result=fake_result)


class TestApiKeyMiddleware:
    """API key 鉴权依赖测试。"""

    def test_no_key_returns_401(self, client_with_key: TestClient) -> None:
        """未携带 API key 的请求应返回 401。"""
        files = {"file": ("test.wav", BytesIO(b"fake-audio"), "audio/wav")}
        resp = client_with_key.post("/v1/audio/transcriptions", files=files, data={"model": "cn_meeting"})
        assert resp.status_code == 401
        assert resp.json()["detail"] == "Invalid API key"

    def test_wrong_key_returns_401(self, client_with_key: TestClient) -> None:
        """携带错误 API key 的请求应返回 401。"""
        files = {"file": ("test.wav", BytesIO(b"fake-audio"), "audio/wav")}
        resp = client_with_key.post(
            "/v1/audio/transcriptions",
            files=files,
            data={"model": "cn_meeting"},
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert resp.status_code == 401

    def test_correct_key_returns_200(self, client_with_key: TestClient) -> None:
        """携带正确 API key 的请求应正常通过。"""
        files = {"file": ("test.wav", BytesIO(b"fake-audio"), "audio/wav")}
        resp = client_with_key.post(
            "/v1/audio/transcriptions",
            files=files,
            data={"model": "cn_meeting"},
            headers={"Authorization": "Bearer test-secret-key"},
        )
        assert resp.status_code == 200

    def test_health_no_key_required(self, client_with_key: TestClient) -> None:
        """/health 端点不需要 API key。"""
        resp = client_with_key.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_no_middleware_when_key_not_set(self, client_no_key: TestClient) -> None:
        """未设置 api_key 时，无需认证即可正常访问。"""
        files = {"file": ("test.wav", BytesIO(b"fake-audio"), "audio/wav")}
        resp = client_no_key.post("/v1/audio/transcriptions", files=files, data={"model": "cn_meeting"})
        assert resp.status_code == 200


class TestLifespan:
    """lifespan 事件测试。"""

    def test_lifespan_logs(self, fake_result: TranscriptionResult) -> None:
        """验证 lifespan 上下文管理器正常执行。"""
        settings = Settings(model_cache_dir=Path("/tmp/test-cache"), device="cpu")
        app = create_app(settings=settings)
        with patch.object(app.state.engine, "transcribe", return_value=fake_result), TestClient(app) as tc:
            resp = tc.get("/health")
            assert resp.status_code == 200
