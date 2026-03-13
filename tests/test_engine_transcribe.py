"""engine.py transcribe / _get_or_load_model / _resolve_model_path 测试。"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from funasr_server.config import Settings
from funasr_server.engine import FunASREngine, FunASRUnavailableError
from funasr_server.model_pool import TTLModelPool


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(model_cache_dir=tmp_path, device="cpu")


@pytest.fixture
def pool() -> TTLModelPool[Any]:
    return TTLModelPool(ttl_seconds=900)


@pytest.fixture
def engine(settings: Settings, pool: TTLModelPool[Any]) -> FunASREngine:
    return FunASREngine(settings=settings, model_pool=pool)


class TestResolveModelPath:
    """_resolve_model_path 测试。"""

    def test_returns_local_path_when_exists(self, engine: FunASREngine, tmp_path: Path) -> None:
        """model_cache_dir 下存在模型目录时，返回本地绝对路径。"""
        model_id = "iic/test_model"
        model_dir = tmp_path / model_id
        model_dir.mkdir(parents=True)

        result = engine._resolve_model_path(model_id)
        assert result == str(model_dir)

    def test_returns_local_path_models_prefix(self, engine: FunASREngine, tmp_path: Path) -> None:
        """models/ 前缀下存在模型目录时，返回本地路径。"""
        model_id = "iic/test_model"
        model_dir = tmp_path / "models" / model_id
        model_dir.mkdir(parents=True)

        result = engine._resolve_model_path(model_id)
        assert result == str(model_dir)

    def test_returns_local_path_hub_prefix(self, engine: FunASREngine, tmp_path: Path) -> None:
        """hub/ 前缀下存在模型目录时，返回本地路径。"""
        model_id = "iic/test_model"
        model_dir = tmp_path / "hub" / model_id
        model_dir.mkdir(parents=True)

        result = engine._resolve_model_path(model_id)
        assert result == str(model_dir)

    def test_returns_model_id_when_not_found(self, engine: FunASREngine) -> None:
        """本地缓存中不存在模型目录时，返回原始 model_id。"""
        model_id = "iic/nonexistent_model"
        result = engine._resolve_model_path(model_id)
        assert result == model_id

    def test_prefers_root_over_models_prefix(self, engine: FunASREngine, tmp_path: Path) -> None:
        """根目录和 models/ 都存在时，优先匹配根目录。"""
        model_id = "iic/test_model"
        (tmp_path / model_id).mkdir(parents=True)
        (tmp_path / "models" / model_id).mkdir(parents=True)

        result = engine._resolve_model_path(model_id)
        assert result == str(tmp_path / model_id)


class TestTranscribe:
    """transcribe 方法完整调用链测试。"""

    def test_transcribe_returns_result(self, engine: FunASREngine, tmp_path: Path) -> None:
        """mock AutoModel 并验证 transcribe 返回正确的 TranscriptionResult。"""
        audio_path = tmp_path / "test.wav"
        audio_path.write_bytes(b"fake-audio")

        mock_model = MagicMock()
        mock_model.generate.return_value = [
            {
                "text": "你好世界",
                "sentence_info": [
                    {"start": 0, "end": 1500, "text": "你好", "spk": "spk_0", "confidence": 0.95},
                    {"start": 1500, "end": 3000, "text": "世界", "spk": "spk_1", "confidence": 0.90},
                ],
            }
        ]

        with patch.object(engine.model_pool, "get_or_create", return_value=mock_model):
            result = engine.transcribe(audio_path, model="cn_meeting", language="zh")

        assert result.text == "你好世界"
        assert result.language == "zh"
        assert result.duration == 3.0
        assert len(result.segments) == 2
        assert result.segments[0].x_speaker_id == "spk_0"
        assert len(result.x_speakers) == 2

    def test_transcribe_with_hotwords(self, engine: FunASREngine, tmp_path: Path) -> None:
        """传入 hotwords 时应在 generate_kwargs 中包含 hotword 参数。"""
        audio_path = tmp_path / "test.wav"
        audio_path.write_bytes(b"fake-audio")

        mock_model = MagicMock()
        mock_model.generate.return_value = [{"text": "hello", "sentence_info": []}]

        with patch.object(engine.model_pool, "get_or_create", return_value=mock_model):
            result = engine.transcribe(audio_path, model="cn_meeting", hotwords="关键词 测试")

        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["hotword"] == "关键词 测试"
        assert result.text == "hello"

    def test_transcribe_no_hotwords_for_unsupported_profile(self, engine: FunASREngine, tmp_path: Path) -> None:
        """对不支持 hotwords 的 profile，即使传入 hotwords 也不会加入 generate_kwargs。"""
        audio_path = tmp_path / "test.wav"
        audio_path.write_bytes(b"fake-audio")

        mock_model = MagicMock()
        mock_model.generate.return_value = [{"text": "hello", "sentence_info": []}]

        with patch.object(engine.model_pool, "get_or_create", return_value=mock_model):
            engine.transcribe(audio_path, model="multilingual_rich", hotwords="some words")

        call_kwargs = mock_model.generate.call_args[1]
        assert "hotword" not in call_kwargs

    def test_transcribe_empty_results_raises(self, engine: FunASREngine, tmp_path: Path) -> None:
        """FunASR 返回空列表时应抛出 RuntimeError。"""
        audio_path = tmp_path / "test.wav"
        audio_path.write_bytes(b"fake-audio")

        mock_model = MagicMock()
        mock_model.generate.return_value = []

        with patch.object(engine.model_pool, "get_or_create", return_value=mock_model):
            with pytest.raises(RuntimeError, match="no transcription results"):
                engine.transcribe(audio_path, model="cn_meeting")

    def test_transcribe_passes_gpu_params(self, tmp_path: Path, pool: TTLModelPool[Any]) -> None:
        """batch_size_s 和 merge_length_s 从 Settings 正确传递到 generate_kwargs。"""
        custom_settings = Settings(model_cache_dir=tmp_path, device="cpu", batch_size_s=200, merge_length_s=30)
        custom_engine = FunASREngine(settings=custom_settings, model_pool=pool)

        audio_path = tmp_path / "test.wav"
        audio_path.write_bytes(b"fake-audio")

        mock_model = MagicMock()
        mock_model.generate.return_value = [{"text": "ok", "sentence_info": []}]

        with patch.object(custom_engine.model_pool, "get_or_create", return_value=mock_model):
            custom_engine.transcribe(audio_path, model="cn_meeting")

        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["batch_size_s"] == 200
        assert call_kwargs["merge_length_s"] == 30

    def test_transcribe_default_language(self, engine: FunASREngine, tmp_path: Path) -> None:
        """不传 language 时使用 profile 默认语言。"""
        audio_path = tmp_path / "test.wav"
        audio_path.write_bytes(b"fake-audio")

        mock_model = MagicMock()
        mock_model.generate.return_value = [{"text": "ok", "sentence_info": []}]

        with patch.object(engine.model_pool, "get_or_create", return_value=mock_model):
            result = engine.transcribe(audio_path, model="cn_meeting")

        assert result.language == "zh"
        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["language"] == "zh"


class TestGetOrLoadModel:
    """_get_or_load_model 测试。"""

    def test_funasr_unavailable_raises(self, engine: FunASREngine) -> None:
        """FunASR 未安装时应抛出 FunASRUnavailableError。"""
        # 清空 pool 确保需要调用 loader
        engine.model_pool._entries.clear()

        with patch.dict("sys.modules", {"funasr": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module named 'funasr'")):
                with pytest.raises(FunASRUnavailableError, match="FunASR is not installed"):
                    engine._get_or_load_model("cn_meeting")

    def test_loads_model_with_all_components(self, engine: FunASREngine, tmp_path: Path) -> None:
        """验证 _get_or_load_model 加载时包含 punc_model 和 spk_model。"""
        engine.model_pool._entries.clear()
        mock_auto_model = MagicMock()

        fake_funasr = MagicMock()
        fake_funasr.AutoModel = mock_auto_model

        with patch.dict("sys.modules", {"funasr": fake_funasr}):
            engine._get_or_load_model("cn_meeting")

        call_kwargs = mock_auto_model.call_args[1]
        assert "model" in call_kwargs
        assert "vad_model" in call_kwargs
        assert "punc_model" in call_kwargs
        assert "spk_model" in call_kwargs
        assert call_kwargs["device"] == "cpu"
        assert call_kwargs["trust_remote_code"] is True

    def test_loads_model_without_punc_model(self, engine: FunASREngine, tmp_path: Path) -> None:
        """multilingual_rich profile 没有 punc_model_id 时不传 punc_model。"""
        engine.model_pool._entries.clear()
        mock_auto_model = MagicMock()

        fake_funasr = MagicMock()
        fake_funasr.AutoModel = mock_auto_model

        with patch.dict("sys.modules", {"funasr": fake_funasr}):
            engine._get_or_load_model("multilingual_rich")

        call_kwargs = mock_auto_model.call_args[1]
        assert "model" in call_kwargs
        assert "vad_model" in call_kwargs
        assert "punc_model" not in call_kwargs
        assert "spk_model" in call_kwargs
