"""SRT / VTT / TXT formatters for transcription results."""

from __future__ import annotations

from funasr_server.schemas import TranscriptionResult


def format_txt(result: TranscriptionResult) -> str:
    return result.text


def _format_timestamp(seconds: float, *, vtt: bool = False) -> str:
    total_ms = int(round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1_000)
    separator = "." if vtt else ","
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def format_srt(result: TranscriptionResult) -> str:
    lines: list[str] = []
    for index, segment in enumerate(result.segments, start=1):
        lines.extend(
            [
                str(index),
                f"{_format_timestamp(segment.start)} --> {_format_timestamp(segment.end)}",
                segment.text,
                "",
            ]
        )
    return "\n".join(lines).strip()


def format_vtt(result: TranscriptionResult) -> str:
    lines = ["WEBVTT", ""]
    for segment in result.segments:
        lines.extend(
            [
                f"{_format_timestamp(segment.start, vtt=True)} --> {_format_timestamp(segment.end, vtt=True)}",
                segment.text,
                "",
            ]
        )
    return "\n".join(lines).strip()
