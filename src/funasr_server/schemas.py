"""OpenAI-compatible response schemas with x_ extensions.

本模块定义了语音转写服务的所有数据模型（Schema）。
采用 Pydantic BaseModel 实现数据校验和序列化，保证 API 响应格式
与 OpenAI Whisper API 兼容。

核心设计思路:
  - 标准字段（text, language, duration, segments）完全对齐 OpenAI Whisper API
  - 扩展字段统一使用 x_ 前缀（如 x_speaker_id, x_speakers），避免与上游字段冲突
  - TranscriptionResult 是引擎内部产物，TranscriptionResponse / VerboseTranscriptionResponse
    是面向调用方的 API 响应体

数据流向:
  engine.py 推理完成 → 构建 TranscriptionResult
    → 如果 response_format 是 json    → 转换为 TranscriptionResponse（仅含 text）
    → 如果 response_format 是 verbose_json → 转换为 VerboseTranscriptionResponse（完整信息）
    → 如果是 srt/vtt/txt             → 交给 formatters.py 处理为纯文本
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Segment(BaseModel):
    """单个语音片段（句子级别）的转写结果。

    每个 Segment 对应音频中的一段连续语音，包含起止时间和转写文字。
    当启用说话人识别（Speaker Diarization）时，还会附带说话人信息。

    与 OpenAI Whisper API 的 segment 结构保持兼容，
    扩展字段以 x_ 开头。
    """

    id: int  # 片段序号，从 0 开始递增，用于排序和定位
    start: float  # 片段起始时间（单位: 秒），与 OpenAI 保持一致
    end: float  # 片段结束时间（单位: 秒）
    text: str  # 该片段的转写文本内容

    # ---- 以下为扩展字段（x_ 前缀）----
    x_speaker_id: str | None = None  # 说话人 ID（如 "spk_0"），未启用说话人识别时为 None
    x_confidence: float | None = None  # 该片段的识别置信度（0.0 ~ 1.0），部分模型可能不提供


class Speaker(BaseModel):
    """说话人统计信息。

    当模型 Profile 支持说话人识别（如 cn_meeting 使用 CAM++ 模型）时，
    引擎会聚合每位说话人的统计数据，放入响应的 x_speakers 数组中。
    """

    speaker_id: str  # 说话人唯一标识（如 "spk_0", "spk_1"）
    display_name: str  # 显示名称，通常与 speaker_id 相同，可由上层业务覆盖
    segment_count: int  # 该说话人的发言片段数量
    total_duration: float  # 该说话人的总发言时长（单位: 秒）


class TranscriptionResult(BaseModel):
    """引擎内部的转写结果，用于在 engine → router 之间传递数据。

    这是一个"胖"模型，包含所有可用信息。Router 层会根据
    调用方请求的 response_format 决定最终输出哪些字段:
      - json         → 只取 text 字段，构建 TranscriptionResponse
      - verbose_json → 取全部字段，构建 VerboseTranscriptionResponse
      - srt/vtt/txt  → 交给 formatters.py 格式化为纯文本
    """

    text: str  # 完整的转写文本（所有片段拼接后的结果）
    language: str  # 检测到的语言代码（如 "zh", "en", "ja"）
    duration: float  # 音频总时长（单位: 秒）
    segments: list[Segment] = Field(default_factory=list)  # 分段转写结果列表
    x_speakers: list[Speaker] = Field(default_factory=list)  # 说话人统计列表（扩展字段）


class TranscriptionResponse(BaseModel):
    """OpenAI-compatible 简洁响应，对应 response_format="json"。

    这是最精简的响应格式，仅包含转写文本。
    与 OpenAI Whisper API 的 JSON 响应格式完全一致。

    示例响应:
      {"text": "你好，欢迎参加今天的会议。"}
    """

    text: str  # 完整的转写文本


class VerboseTranscriptionResponse(BaseModel):
    """OpenAI-compatible 详细响应，对应 response_format="verbose_json"。

    在简洁响应的基础上，增加了语言、时长、分段详情和说话人信息。
    标准字段与 OpenAI Whisper API 兼容，扩展字段以 x_ 为前缀。

    示例响应:
      {
        "text": "你好，欢迎参加今天的会议。",
        "language": "zh",
        "duration": 12.5,
        "segments": [...],
        "x_speakers": [...]
      }
    """

    text: str  # 完整的转写文本
    language: str  # 检测到的语言代码
    duration: float  # 音频总时长（单位: 秒）
    segments: list[Segment] = Field(default_factory=list)  # 分段转写结果列表
    x_speakers: list[Speaker] = Field(default_factory=list)  # 说话人统计列表（扩展字段）
