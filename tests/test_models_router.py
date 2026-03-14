"""GET /v1/models 端点测试。"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from funasr_server.app import create_app
from funasr_server.config import Settings


def test_list_models() -> None:
    """GET /v1/models 返回 OpenAI 兼容的模型列表。"""
    settings = Settings(model_cache_dir=Path("/tmp/test-cache"), device="cpu")
    app = create_app(settings=settings)
    client = TestClient(app)

    resp = client.get("/v1/models")
    assert resp.status_code == 200

    body = resp.json()
    assert body["object"] == "list"
    assert len(body["data"]) >= 2

    model_ids = {m["id"] for m in body["data"]}
    assert "cn_meeting" in model_ids
    assert "multilingual_rich" in model_ids

    for m in body["data"]:
        assert m["object"] == "model"
        assert m["owned_by"] == "funasr-server"


def test_list_models_with_api_key() -> None:
    """启用 API key 时，GET /v1/models 需要鉴权。"""
    settings = Settings(model_cache_dir=Path("/tmp/test-cache"), device="cpu", api_key="secret")
    app = create_app(settings=settings)
    client = TestClient(app)

    # 无 key → 401
    resp = client.get("/v1/models")
    assert resp.status_code == 401

    # 有 key → 200
    resp = client.get("/v1/models", headers={"Authorization": "Bearer secret"})
    assert resp.status_code == 200
    assert resp.json()["object"] == "list"
