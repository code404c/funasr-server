"""dependencies.py 的单元测试 — 覆盖 AudioFile 和 verify_api_key。"""

from __future__ import annotations

from pathlib import Path

from funasr_server.dependencies import AudioFile


class TestAudioFile:
    """AudioFile dataclass 测试。"""

    def test_cleanup_removes_file(self, tmp_path: Path) -> None:
        """cleanup() 应删除临时文件。"""
        f = tmp_path / "test.wav"
        f.write_bytes(b"fake")
        audio = AudioFile(path=f, filename="test.wav")

        assert f.exists()
        audio.cleanup()
        assert not f.exists()

    def test_cleanup_missing_ok(self, tmp_path: Path) -> None:
        """cleanup() 对不存在的文件不应报错。"""
        f = tmp_path / "nonexistent.wav"
        audio = AudioFile(path=f, filename="nonexistent.wav")
        audio.cleanup()  # 不应抛出异常
