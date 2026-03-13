"""FunASR 推理核心 — 无 minutes 依赖。"""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Any

from loguru import logger

from funasr_server.config import Settings
from funasr_server.model_pool import TTLModelPool
from funasr_server.profiles import get_profile_spec
from funasr_server.schemas import Segment, Speaker, TranscriptionResult


class FunASRUnavailableError(RuntimeError):
    pass


class FunASREngine:
    def __init__(self, *, settings: Settings, model_pool: TTLModelPool[Any]) -> None:
        self.settings = settings
        self.model_pool = model_pool

    def transcribe(
        self,
        audio_path: Path,
        *,
        model: str = "cn_meeting",
        language: str | None = None,
        hotwords: str | None = None,
    ) -> TranscriptionResult:
        profile = get_profile_spec(model)
        asr_model = self._get_or_load_model(profile.name.value)

        generate_kwargs: dict[str, Any] = {
            "input": str(audio_path),
            "cache": {},
            "language": language or profile.default_language,
            "use_itn": True,
            "batch_size_s": 60,
            "merge_vad": True,
            "merge_length_s": 15,
        }
        if hotwords and profile.supports_hotwords:
            generate_kwargs["hotword"] = hotwords

        logger.debug("Starting inference for profile={} file={}", profile.name.value, audio_path.name)
        t0 = time.monotonic()

        results = asr_model.generate(**generate_kwargs)
        if not results:
            raise RuntimeError("FunASR returned no transcription results.")

        item = results[0]
        full_text = str(item.get("text", "")).strip()
        sentence_info = item.get("sentence_info") or []
        lang = language or profile.default_language
        segments = self._build_segments(sentence_info, full_text)
        speakers = self._build_speakers(segments)
        duration = self._compute_duration(segments)

        elapsed = time.monotonic() - t0
        logger.info(
            "Transcription completed: profile={} file={} segments={} speakers={} duration={:.3f}s elapsed={:.3f}s",
            profile.name.value,
            audio_path.name,
            len(segments),
            len(speakers),
            duration,
            elapsed,
        )

        return TranscriptionResult(
            text=full_text,
            language=lang,
            duration=duration,
            segments=segments,
            x_speakers=speakers,
        )

    def _get_or_load_model(self, cache_key: str):
        def _loader():
            try:
                from funasr import AutoModel
            except ImportError as exc:
                raise FunASRUnavailableError(
                    "FunASR is not installed. Install the project dependencies to enable transcription."
                ) from exc

            logger.info("Loading model: cache_key={} device={}", cache_key, self.settings.device)
            t0 = time.monotonic()

            profile = get_profile_spec(cache_key)
            model_kwargs: dict[str, Any] = {
                "model": self._resolve_model_path(profile.asr_model_id),
                "vad_model": self._resolve_model_path(profile.vad_model_id),
                "device": self.settings.device,
                "trust_remote_code": True,
            }
            if profile.punc_model_id:
                model_kwargs["punc_model"] = self._resolve_model_path(profile.punc_model_id)
            if profile.speaker_model_id:
                model_kwargs["spk_model"] = self._resolve_model_path(profile.speaker_model_id)
            result = AutoModel(**model_kwargs)

            elapsed = time.monotonic() - t0
            logger.info("Model loaded: cache_key={} elapsed={:.3f}s", cache_key, elapsed)
            return result

        return self.model_pool.get_or_create(cache_key, _loader)

    def _resolve_model_path(self, model_id: str) -> str:
        """如果 model_cache_dir 下存在对应目录，返回本地路径；否则返回原始 model ID 走 ModelScope 下载。"""
        cache = self.settings.model_cache_dir.expanduser()
        for prefix in ("", "models/", "hub/"):
            candidate = cache / prefix / model_id
            if candidate.is_dir():
                logger.debug("Resolved model path: {} -> {}", model_id, candidate)
                return str(candidate)
        logger.debug("No local cache for model {}, will download from ModelScope", model_id)
        return model_id

    @staticmethod
    def _build_segments(sentence_info: list[dict[str, Any]], fallback_text: str) -> list[Segment]:
        if not sentence_info:
            return [
                Segment(
                    id=0,
                    start=0.0,
                    end=1.0,
                    text=fallback_text,
                    x_speaker_id=None,
                    x_confidence=None,
                )
            ]

        segments: list[Segment] = []
        for index, item in enumerate(sentence_info):
            speaker_raw = item.get("spk") or item.get("speaker")
            speaker_id = str(speaker_raw) if speaker_raw is not None else None
            start_ms = int(item.get("start", 0))
            end_ms = int(item.get("end", start_ms + 1000))
            segments.append(
                Segment(
                    id=index,
                    start=start_ms / 1000.0,
                    end=end_ms / 1000.0,
                    text=str(item.get("text", "")).strip(),
                    x_speaker_id=speaker_id,
                    x_confidence=item.get("confidence"),
                )
            )
        return segments

    @staticmethod
    def _build_speakers(segments: list[Segment]) -> list[Speaker]:
        totals: Counter[str] = Counter()
        counts: Counter[str] = Counter()
        for seg in segments:
            if seg.x_speaker_id is None:
                continue
            totals[seg.x_speaker_id] += seg.end - seg.start
            counts[seg.x_speaker_id] += 1
        return [
            Speaker(
                speaker_id=sid,
                display_name=sid.replace("_", " ").title(),
                segment_count=counts[sid],
                total_duration=round(totals[sid], 3),
            )
            for sid in sorted(counts)
        ]

    @staticmethod
    def _compute_duration(segments: list[Segment]) -> float:
        if not segments:
            return 0.0
        return max(seg.end for seg in segments)
