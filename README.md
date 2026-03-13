# funasr-server

OpenAI-compatible STT server powered by FunASR.

Exposes `POST /v1/audio/transcriptions` with the same API contract as OpenAI Whisper,
plus `x_` extensions for speaker diarization and confidence scores.

## Features

- OpenAI Whisper API 兼容（可直接用 OpenAI SDK 调用）
- 多模型 profile 支持（中文会议、多语种）
- 说话人识别（speaker diarization）
- 热词增强（hotwords）
- GPU 推理参数可调（`batch_size_s` 等，按 GPU 型号优化）
- TTL 模型池，自动释放闲置 GPU 显存
- 可选 API Key 鉴权
- 多种输出格式：json / verbose_json / text / srt / vtt

## Quick Start

```bash
make install
make download-models
make dev
```

## Docker 部署

### 构建镜像

```bash
# 默认使用 nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04 基础镜像
make docker-build

# 自定义基础镜像
docker build -f docker/Dockerfile \
  --build-arg BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 \
  -t funasr-server:latest .
```

### GPU 预设

项目提供了针对不同 GPU 的环境变量预设文件：

```bash
# RTX A4000 16GB
cp .env.a4000 .env

# H100 80GB
cp .env.h100 .env
```

| GPU 型号 | VRAM | 推荐 `batch_size_s` | 预设文件 |
|---------|------|---------------------|---------|
| RTX A4000 | 16 GB | 60 | `.env.a4000` |
| RTX 4090 | 24 GB | 120 | — |
| A100 | 40/80 GB | 200–300 | — |
| H100 | 80 GB | 300 | `.env.h100` |

> `batch_size_s` 是 FunASR VAD 切分后每批送入 ASR 模型的最大音频秒数，
> 显存越大可设越高，提升长音频吞吐量。与 CTranslate2 的 `batch_size`（token 数）含义不同。

### 启动服务

```bash
# 使用 docker compose（推荐）
docker compose up -d

# 验证健康检查
curl http://localhost:8101/health

# 查看日志
make docker-logs
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `FUNASR_MODEL_CACHE_DIR` | `/modelscope-cache` | 模型缓存目录（本地开发用 `~/models`） |
| `FUNASR_DEVICE` | `cuda:0` | PyTorch 推理设备 |
| `FUNASR_MODEL_TTL_SECONDS` | `900` | 模型池 TTL（秒），-1 表示永不过期 |
| `FUNASR_BATCH_SIZE_S` | `60` | VAD 切分后每批最大秒数，按 GPU 显存调整 |
| `FUNASR_MERGE_LENGTH_S` | `15` | VAD 合并后每段最大秒数 |
| `FUNASR_API_KEY` | (空) | 可选 Bearer token 认证 |
| `FUNASR_LOG_LEVEL` | `INFO` | 日志级别 |
| `FUNASR_LOG_JSON` | `true` | 是否输出 JSON 格式日志 |

## API

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@meeting.wav" \
  -F "model=cn_meeting" \
  -F "response_format=verbose_json"
```

## 模型 Profile

| model 参数 | ASR 引擎 | 语言 | 热词 | 说话人识别 |
|-----------|---------|------|------|-----------|
| `cn_meeting` | Paraformer-large + FSMN-VAD + CT-Punc + CAM++ | 中文 | 支持 | 支持 |
| `multilingual_rich` | SenseVoice + FSMN-VAD + CAM++ | 自动 | 不支持 | 支持 |
