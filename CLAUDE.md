# funasr-server 项目指南

公司级 OpenAI-compatible STT 微服务，基于 FunASR 引擎提供语音转文字能力。

## API 合约

```
POST /v1/audio/transcriptions
Content-Type: multipart/form-data

标准字段（OpenAI 兼容）:
  file:              UploadFile (必需)
  model:             str (default="cn_meeting") → 映射到内部 ProfileSpec
  language:          str | None
  response_format:   "json" | "verbose_json" | "text" | "srt" | "vtt"

扩展字段:
  hotwords:          str | None (逗号分隔热词)

响应 (verbose_json):
  text, language, duration, segments[], x_speakers[]
  segments 中 start/end 使用秒（与 OpenAI 一致）
  扩展字段统一用 x_ 前缀
```

## Call Flow

```
POST /v1/audio/transcriptions
  → routers/transcriptions.py (multipart 解析, 临时文件写入)
  → engine.py::FunASREngine.transcribe (模型加载 + 推理)
  → model_pool.py::TTLModelPool (线程安全 TTL 模型缓存)
  → profiles.py (model → ProfileSpec 映射)
  → schemas.py (构建 OpenAI-compatible 响应)
  → formatters.py (srt/vtt/txt 格式化, 仅非 JSON 格式时)
```

## 工具链

| 包管理器 | 测试框架 | Linter/Formatter | 配置文件 |
|---------|---------|-----------------|---------|
| **uv** | **pytest** | **ruff** | `pyproject.toml` + `uv.lock` |

常用命令：
- 安装依赖: `make install`
- 启动开发服务器: `make dev`
- 格式化: `make format`
- 检查: `make lint`
- 全量检查: `make check`
- 测试: `make test`
- 下载模型: `make download-models`

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `FUNASR_MODEL_CACHE_DIR` | `/modelscope-cache` | 模型缓存目录（本地开发用 `~/models`） |
| `FUNASR_DEVICE` | `cuda:0` | PyTorch 推理设备 |
| `FUNASR_MODEL_TTL_SECONDS` | `900` | 模型池 TTL（秒），-1 表示永不过期 |
| `FUNASR_API_KEY` | (空) | 可选 Bearer token 认证 |
| `FUNASR_LOG_LEVEL` | `INFO` | 日志级别 |
| `FUNASR_LOG_JSON` | `true` | 是否输出 JSON 格式日志 |

## 模型 Profile

| model 参数 | ASR 引擎 | 语言 | 热词 | 说话人识别 |
|-----------|---------|------|------|-----------|
| `cn_meeting` | Paraformer-large + FSMN-VAD + CT-Punc + CAM++ | 中文 | 支持 | 支持 |
| `multilingual_rich` | SenseVoice + FSMN-VAD + CAM++ | 自动 | 不支持 | 支持 |

模型存放路径格式: `{FUNASR_MODEL_CACHE_DIR}/iic/{model_name}`

## 调用方

- **minutes** (`~/workspaces/app/minutes/`): 会议转写后端，通过 `MINUTES_STT_BASE_URL` 指向本服务
- 其他项目可直接用 OpenAI SDK 或 curl 调用，无需特殊客户端

## 代码规范

- 注释、文档字符串: 简体中文
- 日志、代码内文本: 英文
- API 合约变更必须同步通知所有调用方
- 新增 model profile 时，同步更新 `profiles.py` 和本文件

## 提交前自检

- 运行 `make check`（format + lint + test）
- API 合约变更时：确认 response schema 仍兼容 OpenAI Whisper API
- 新增/修改 profile 时：确认 `scripts/download_models.py` 能正确下载对应模型
