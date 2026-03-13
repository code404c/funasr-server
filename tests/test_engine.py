"""engine.py 的单元测试 — 不依赖真实 FunASR 模型。"""

from __future__ import annotations

from funasr_server.engine import FunASREngine
from funasr_server.schemas import Segment


def test_build_segments_from_sentence_info() -> None:
    sentence_info = [
        {"start": 0, "end": 1500, "text": "你好", "spk": "speaker_1", "confidence": 0.95},
        {"start": 1500, "end": 3000, "text": "世界", "spk": "speaker_2", "confidence": 0.88},
    ]
    segments = FunASREngine._build_segments(sentence_info, "你好世界")

    assert len(segments) == 2
    assert segments[0].start == 0.0
    assert segments[0].end == 1.5
    assert segments[0].text == "你好"
    assert segments[0].x_speaker_id == "speaker_1"
    assert segments[0].x_confidence == 0.95
    assert segments[1].start == 1.5
    assert segments[1].end == 3.0


def test_build_segments_fallback_when_empty() -> None:
    segments = FunASREngine._build_segments([], "hello world")

    assert len(segments) == 1
    assert segments[0].text == "hello world"
    assert segments[0].start == 0.0
    assert segments[0].end == 1.0
    assert segments[0].x_speaker_id is None


def test_build_speakers() -> None:
    segments = [
        Segment(id=0, start=0.0, end=1.5, text="A", x_speaker_id="speaker_1"),
        Segment(id=1, start=1.5, end=3.0, text="B", x_speaker_id="speaker_2"),
        Segment(id=2, start=3.0, end=5.0, text="C", x_speaker_id="speaker_1"),
    ]
    speakers = FunASREngine._build_speakers(segments)

    assert len(speakers) == 2
    speaker_map = {s.speaker_id: s for s in speakers}
    assert speaker_map["speaker_1"].segment_count == 2
    assert speaker_map["speaker_1"].total_duration == 3.5
    assert speaker_map["speaker_2"].segment_count == 1


def test_compute_duration() -> None:
    segments = [
        Segment(id=0, start=0.0, end=1.5, text="A"),
        Segment(id=1, start=1.5, end=5.2, text="B"),
    ]
    duration = FunASREngine._compute_duration(segments)
    assert duration == 5.2


def test_compute_duration_empty() -> None:
    assert FunASREngine._compute_duration([]) == 0.0
