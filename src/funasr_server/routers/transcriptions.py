"""语音转写 API 路由模块 — 提供 OpenAI Whisper API 兼容的转写端点。

该模块实现了 ``POST /v1/audio/transcriptions`` 端点，接收音频文件并返回
转写结果。完整调用链路：

    客户端上传音频 → AudioFileDependency 落盘临时文件
    → engine.transcribe() 执行 FunASR 推理
    → schemas.py 构建响应 → formatters.py 格式化（仅非 JSON 格式时）

支持的响应格式：
    - ``json``：仅返回转写文本（OpenAI 标准）
    - ``verbose_json``：返回文本 + 分段 + 说话人等详细信息（带 x_ 扩展字段）
    - ``text``：纯文本
    - ``srt``：SubRip 字幕格式
    - ``vtt``：WebVTT 字幕格式
"""

from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, Form
from fastapi.responses import PlainTextResponse
from loguru import logger

from funasr_server.dependencies import AudioFileDependency, EngineDependency
from funasr_server.engine import FunASRUnavailableError
from funasr_server.errors import TranscriptionError
from funasr_server.formatters import format_srt, format_txt, format_vtt
from funasr_server.schemas import TranscriptionResponse, VerboseTranscriptionResponse

# 创建路由器，所有端点挂载在 /v1/audio 前缀下
router = APIRouter(prefix="/v1/audio", tags=["transcriptions"])

# 允许的响应格式白名单
ALLOWED_RESPONSE_FORMATS = {"json", "verbose_json", "text", "srt", "vtt"}


@router.post("/transcriptions")
async def create_transcription(
    engine: EngineDependency,
    audio: AudioFileDependency,
    model: Annotated[str, Form()] = "cn_meeting",
    language: Annotated[str | None, Form()] = None,
    response_format: Annotated[str, Form()] = "json",
    hotwords: Annotated[str | None, Form()] = None,
):
    """处理音频转写请求，兼容 OpenAI Whisper ``/v1/audio/transcriptions`` API。

    Args:
        engine: FunASR 推理引擎（通过依赖注入）。
        audio: 已落盘的音频临时文件（通过依赖注入，请求结束后自动清理）。
        model: 模型 profile 名称，默认 ``"cn_meeting"``。
        language: 指定语言代码，为 None 时使用 profile 默认语言。
        response_format: 响应格式，默认 ``"json"``。
        hotwords: 热词列表，逗号分隔。

    Returns:
        根据 response_format 返回不同格式的响应。

    Raises:
        TranscriptionError: 转写过程中的各类错误。
    """
    if response_format not in ALLOWED_RESPONSE_FORMATS:
        raise TranscriptionError(
            f"Unsupported response_format: {response_format}",
            status_code=400,
            error_type="invalid_request_error",
        )

    try:
        logger.info("Transcription request: model={} language={} format={}", model, language, response_format)

        t0 = time.perf_counter()
        result = engine.transcribe(audio.path, model=model, language=language, hotwords=hotwords)
        duration_ms = (time.perf_counter() - t0) * 1000
        logger.info("Transcription completed in {:.0f}ms", duration_ms)

    except FunASRUnavailableError as exc:
        raise TranscriptionError(
            str(exc),
            status_code=503,
            error_type="service_unavailable",
        ) from exc
    except ValueError as exc:
        raise TranscriptionError(
            str(exc),
            status_code=400,
            error_type="invalid_request_error",
        ) from exc
    except Exception as exc:
        logger.exception("Transcription failed")
        raise TranscriptionError(
            "Internal transcription error",
            status_code=500,
            error_type="internal_error",
        ) from exc

    # ── 根据 response_format 构造并返回不同格式的响应 ──

    if response_format == "json":
        return TranscriptionResponse(text=result.text)

    if response_format == "verbose_json":
        return VerboseTranscriptionResponse(
            text=result.text,
            language=result.language,
            duration=result.duration,
            segments=result.segments,
            x_speakers=result.x_speakers,
        )

    if response_format == "text":
        return PlainTextResponse(format_txt(result))

    if response_format == "srt":
        return PlainTextResponse(format_srt(result))

    if response_format == "vtt":
        return PlainTextResponse(format_vtt(result))
