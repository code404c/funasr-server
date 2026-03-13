"""profiles.py 的单元测试 — 覆盖 resolve_profile 和 get_profile_spec。"""

from __future__ import annotations

import pytest

from funasr_server.profiles import ModelProfile, ProfileSpec, get_profile_spec, resolve_profile


class TestResolveProfile:
    """resolve_profile 函数测试。"""

    def test_none_returns_default(self) -> None:
        """传入 None 时返回默认的 CN_MEETING。"""
        result = resolve_profile(None)
        assert result == ModelProfile.CN_MEETING

    def test_string_returns_profile(self) -> None:
        """传入字符串 'cn_meeting' 时返回对应 ModelProfile。"""
        result = resolve_profile("cn_meeting")
        assert result == ModelProfile.CN_MEETING

    def test_string_multilingual(self) -> None:
        """传入字符串 'multilingual_rich' 时返回对应 ModelProfile。"""
        result = resolve_profile("multilingual_rich")
        assert result == ModelProfile.MULTILINGUAL_RICH

    def test_model_profile_passthrough(self) -> None:
        """传入 ModelProfile 枚举值时直接返回。"""
        result = resolve_profile(ModelProfile.MULTILINGUAL_RICH)
        assert result == ModelProfile.MULTILINGUAL_RICH

    def test_invalid_string_raises(self) -> None:
        """传入无效字符串时应抛出 ValueError。"""
        with pytest.raises(ValueError, match="not_a_profile"):
            resolve_profile("not_a_profile")


class TestGetProfileSpec:
    """get_profile_spec 函数测试。"""

    def test_returns_spec_for_cn_meeting(self) -> None:
        """获取 cn_meeting 的 ProfileSpec。"""
        spec = get_profile_spec("cn_meeting")
        assert isinstance(spec, ProfileSpec)
        assert spec.name == ModelProfile.CN_MEETING
        assert spec.default_language == "zh"
        assert spec.supports_hotwords is True
        assert spec.punc_model_id is not None

    def test_returns_spec_for_multilingual(self) -> None:
        """获取 multilingual_rich 的 ProfileSpec。"""
        spec = get_profile_spec("multilingual_rich")
        assert isinstance(spec, ProfileSpec)
        assert spec.name == ModelProfile.MULTILINGUAL_RICH
        assert spec.default_language == "auto"
        assert spec.supports_hotwords is False
        assert spec.punc_model_id is None

    def test_returns_spec_for_none(self) -> None:
        """传入 None 时返回默认 profile 的 spec。"""
        spec = get_profile_spec(None)
        assert spec.name == ModelProfile.CN_MEETING

    def test_returns_spec_for_enum(self) -> None:
        """直接传入 ModelProfile 枚举值。"""
        spec = get_profile_spec(ModelProfile.CN_MEETING)
        assert spec.name == ModelProfile.CN_MEETING
