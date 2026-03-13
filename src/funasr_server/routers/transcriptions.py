"""POST /v1/audio/transcriptions — OpenAI-compatible endpoint."""

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

router = APIRouter(prefix="/v1/audio", tags=["transcriptions"])

ALLOWED_RESPONSE_FORMATS = {"json", "verbose_json", "text", "srt", "vtt"}


def _get_engine(request: Request) -> FunASREngine:
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
    if response_format not in ALLOWED_RESPONSE_FORMATS:
        raise HTTPException(status_code=400, detail=f"Unsupported response_format: {response_format}")

    engine = _get_engine(request)

    suffix = Path(file.filename).suffix if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        content = await file.read()
        tmp_path.write_bytes(content)

        logger.info("Transcription request: model={} language={} format={}", model, language, response_format)
        t0 = time.perf_counter()
        result = engine.transcribe(tmp_path, model=model, language=language, hotwords=hotwords)
        duration_ms = (time.perf_counter() - t0) * 1000
        logger.info("Transcription completed in {:.0f}ms", duration_ms)
    except FunASRUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail="Internal transcription error") from exc
    finally:
        tmp_path.unlink(missing_ok=True)

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
