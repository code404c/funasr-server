"""API 端点测试 — 使用 fake engine 避免 FunASR 依赖。"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from funasr_server.app import create_app
from funasr_server.config import Settings
from funasr_server.schemas import Segment, Speaker, TranscriptionResult


@pytest.fixture
def fake_result() -> TranscriptionResult:
    return TranscriptionResult(
        text="你好世界",
        language="zh",
        duration=3.0,
        segments=[
            Segment(id=0, start=0.0, end=1.5, text="你好", x_speaker_id="speaker_1", x_confidence=0.95),
            Segment(id=1, start=1.5, end=3.0, text="世界", x_speaker_id="speaker_2", x_confidence=0.88),
        ],
        x_speakers=[
            Speaker(speaker_id="speaker_1", display_name="Speaker 1", segment_count=1, total_duration=1.5),
            Speaker(speaker_id="speaker_2", display_name="Speaker 2", segment_count=1, total_duration=1.5),
        ],
    )


@pytest.fixture
def client(fake_result: TranscriptionResult) -> TestClient:
    settings = Settings(model_cache_dir=Path("/tmp/test-cache"), device="cpu")
    app = create_app(settings=settings)

    with patch.object(app.state.engine, "transcribe", return_value=fake_result):
        yield TestClient(app)


def _upload(client: TestClient, **kwargs) -> ...:
    defaults = {"model": "cn_meeting", "response_format": "json"}
    defaults.update(kwargs)
    files = {"file": ("test.wav", BytesIO(b"fake-audio"), "audio/wav")}
    return client.post("/v1/audio/transcriptions", files=files, data=defaults)


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_transcription_json(client: TestClient) -> None:
    resp = _upload(client, response_format="json")
    assert resp.status_code == 200
    body = resp.json()
    assert body["text"] == "你好世界"
    assert "segments" not in body


def test_transcription_verbose_json(client: TestClient) -> None:
    resp = _upload(client, response_format="verbose_json")
    assert resp.status_code == 200
    body = resp.json()
    assert body["text"] == "你好世界"
    assert body["language"] == "zh"
    assert body["duration"] == 3.0
    assert len(body["segments"]) == 2
    assert body["segments"][0]["x_speaker_id"] == "speaker_1"
    assert len(body["x_speakers"]) == 2


def test_transcription_text(client: TestClient) -> None:
    resp = _upload(client, response_format="text")
    assert resp.status_code == 200
    assert resp.text == "你好世界"


def test_transcription_srt(client: TestClient) -> None:
    resp = _upload(client, response_format="srt")
    assert resp.status_code == 200
    assert "00:00:00,000 --> 00:00:01,500" in resp.text
    assert "你好" in resp.text


def test_transcription_vtt(client: TestClient) -> None:
    resp = _upload(client, response_format="vtt")
    assert resp.status_code == 200
    assert resp.text.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:01.500" in resp.text


def test_invalid_response_format(client: TestClient) -> None:
    resp = _upload(client, response_format="invalid")
    assert resp.status_code == 400
