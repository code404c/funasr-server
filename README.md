# funasr-server

OpenAI-compatible STT server powered by FunASR.

Exposes `POST /v1/audio/transcriptions` with the same API contract as OpenAI Whisper,
plus `x_` extensions for speaker diarization and confidence scores.

## Quick Start

```bash
make install
make download-models
make dev
```

## API

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@meeting.wav" \
  -F "model=cn_meeting" \
  -F "response_format=verbose_json"
```
