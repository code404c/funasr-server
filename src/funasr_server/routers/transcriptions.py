"""语音转写 API 路由模块 — 提供 OpenAI Whisper API 兼容的转写端点。

该模块实现了 ``POST /v1/audio/transcriptions`` 端点，接收音频文件并返回
转写结果。完整调用链路：

    客户端上传音频 → 本模块解析 multipart 表单 → 写入临时文件
    → engine.py 执行 FunASR 推理 → schemas.py 构建响应
    → formatters.py 格式化为 srt/vtt/txt（仅非 JSON 格式时）

支持的响应格式：
    - ``json``：仅返回转写文本（OpenAI 标准）
    - ``verbose_json``：返回文本 + 分段 + 说话人等详细信息（带 x_ 扩展字段）
    - ``text``：纯文本
    - ``srt``：SubRip 字幕格式
    - ``vtt``：WebVTT 字幕格式
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Form, HTTPException, Request, UploadFile
from fastapi.responses import PlainTextResponse
from loguru import logger

from funasr_server.engine import FunASREngine, FunASRUnavailableError
from funasr_server.formatters import format_srt, format_txt, format_vtt
from funasr_server.schemas import TranscriptionResponse, VerboseTranscriptionResponse

# 创建路由器，所有端点挂载在 /v1/audio 前缀下，OpenAPI 文档中归类到 "transcriptions" 标签
router = APIRouter(prefix="/v1/audio", tags=["transcriptions"])

# 允许的响应格式白名单，用于请求参数校验
ALLOWED_RESPONSE_FORMATS = {"json", "verbose_json", "text", "srt", "vtt"}


def _get_engine(request: Request) -> FunASREngine:
    """从 FastAPI 应用状态中获取 FunASR 推理引擎实例。

    引擎实例在应用启动时（lifespan）挂载到 ``app.state.engine``，
    整个应用生命周期内共享同一个实例。

    Args:
        request: FastAPI 请求对象，通过它可访问 ``app.state``。

    Returns:
        FunASREngine: 全局共享的推理引擎实例。
    """
    return request.app.state.engine


@router.post("/transcriptions")
async def create_transcription(
    request: Request,
    file: UploadFile,
    model: Annotated[str, Form()] = "cn_meeting",
    language: Annotated[str | None, Form()] = None,
    response_format: Annotated[str, Form()] = "json",
    hotwords: Annotated[str | None, Form()] = None,
):
    """处理音频转写请求，兼容 OpenAI Whisper ``/v1/audio/transcriptions`` API。

    该端点接收 multipart/form-data 格式的音频文件，通过 FunASR 引擎进行
    语音识别，并按照指定格式返回转写结果。

    Args:
        request: FastAPI 请求对象，用于获取全局引擎实例。
        file: 上传的音频文件（必需），支持常见音频格式（wav, mp3, m4a 等）。
        model: 模型 profile 名称，映射到 ``profiles.py`` 中的 ProfileSpec。
            默认 ``"cn_meeting"``（中文会议场景，含 VAD + 标点 + 说话人识别）。
        language: 指定语言代码（如 ``"zh"``, ``"en"``）。为 None 时使用
            profile 的默认语言。
        response_format: 响应格式，可选 ``"json"``、``"verbose_json"``、
            ``"text"``、``"srt"``、``"vtt"``。默认 ``"json"``。
        hotwords: 热词列表，逗号分隔（如 ``"张三,人工智能"``），用于提升
            特定词汇的识别准确率。仅部分 profile 支持。

    Returns:
        - ``json`` 格式：返回 ``TranscriptionResponse``（仅包含 text 字段）
        - ``verbose_json`` 格式：返回 ``VerboseTranscriptionResponse``
          （包含 text, language, duration, segments, x_speakers）
        - ``text/srt/vtt`` 格式：返回 ``PlainTextResponse`` 纯文本

    Raises:
        HTTPException(400): response_format 不在白名单内，或请求参数无效。
        HTTPException(503): FunASR 引擎不可用（如依赖未安装）。
        HTTPException(500): 推理过程中发生未预期的内部错误。
    """
    # 校验 response_format 是否在允许列表中，不合法则立即返回 400
    if response_format not in ALLOWED_RESPONSE_FORMATS:
        raise HTTPException(status_code=400, detail=f"Unsupported response_format: {response_format}")

    # 从应用状态中取出全局共享的推理引擎
    engine = _get_engine(request)

    # 保留原始文件的后缀名（如 .wav, .mp3），以便 FunASR 正确识别音频格式；
    # 如果上传文件没有文件名，则默认使用 .wav
    suffix = Path(file.filename).suffix if file.filename else ".wav"

    # 创建临时文件并立即关闭（仅获取路径），设置 delete=False 防止 with 块结束时自动删除；
    # 后续在 finally 中手动清理，确保无论成功或失败都不会泄漏临时文件
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        # 异步读取上传文件的全部字节内容，然后同步写入临时文件；
        # FunASR 引擎需要磁盘上的文件路径作为输入，因此必须先落盘
        content = await file.read()
        tmp_path.write_bytes(content)

        logger.info("Transcription request: model={} language={} format={}", model, language, response_format)

        # 使用高精度计时器记录推理耗时，perf_counter 适合测量短时间间隔
        t0 = time.perf_counter()
        # 调用引擎执行转写，返回 TranscriptionResult（包含文本、分段、说话人等信息）
        result = engine.transcribe(tmp_path, model=model, language=language, hotwords=hotwords)
        duration_ms = (time.perf_counter() - t0) * 1000  # 转换为毫秒
        logger.info("Transcription completed in {:.0f}ms", duration_ms)

    except FunASRUnavailableError as exc:
        # FunASR 依赖未安装或引擎初始化失败 → 503 Service Unavailable
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        # 参数校验失败（如无效的 model 名称）→ 400 Bad Request
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        # 兜底：捕获所有未预期异常，记录完整堆栈后返回 500
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail="Internal transcription error") from exc
    finally:
        # 无论转写成功还是失败，都确保删除临时音频文件，防止磁盘泄漏；
        # missing_ok=True 表示文件若已不存在（极端情况）也不报错
        tmp_path.unlink(missing_ok=True)

    # ── 根据 response_format 构造并返回不同格式的响应 ──

    if response_format == "json":
        # OpenAI 标准简洁响应：仅包含转写文本
        return TranscriptionResponse(text=result.text)

    if response_format == "verbose_json":
        # 详细 JSON 响应：包含文本、语言、时长、逐句分段、说话人列表
        # x_speakers 是本服务的扩展字段（x_ 前缀表示非 OpenAI 标准）
        return VerboseTranscriptionResponse(
            text=result.text,
            language=result.language,
            duration=result.duration,
            segments=result.segments,
            x_speakers=result.x_speakers,
        )

    if response_format == "text":
        # 纯文本格式：将转写结果格式化为无时间戳的纯文本
        return PlainTextResponse(format_txt(result))

    if response_format == "srt":
        # SubRip 字幕格式：带序号和时间戳的标准 .srt 格式
        return PlainTextResponse(format_srt(result))

    if response_format == "vtt":
        # WebVTT 字幕格式：Web 端常用的字幕格式，带 WEBVTT 头部
        return PlainTextResponse(format_vtt(result))
