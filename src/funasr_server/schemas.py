"""OpenAI-compatible response schemas with x_ extensions."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Segment(BaseModel):
    id: int
    start: float
    end: float
    text: str
    x_speaker_id: str | None = None
    x_confidence: float | None = None


class Speaker(BaseModel):
    speaker_id: str
    display_name: str
    segment_count: int
    total_duration: float


class TranscriptionResult(BaseModel):
    """Internal result from the engine, used to build API responses."""

    text: str
    language: str
    duration: float
    segments: list[Segment] = Field(default_factory=list)
    x_speakers: list[Speaker] = Field(default_factory=list)


class TranscriptionResponse(BaseModel):
    """OpenAI-compatible response_format=json."""

    text: str


class VerboseTranscriptionResponse(BaseModel):
    """OpenAI-compatible response_format=verbose_json with x_ extensions."""

    text: str
    language: str
    duration: float
    segments: list[Segment] = Field(default_factory=list)
    x_speakers: list[Speaker] = Field(default_factory=list)
