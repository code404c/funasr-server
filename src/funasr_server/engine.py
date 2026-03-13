"""FunASR 推理引擎模块。

本模块是语音转文字服务的核心推理层，负责：
1. 接收音频文件路径，调用 FunASR 模型进行语音识别（ASR）
2. 将 FunASR 原始推理结果转换为 OpenAI Whisper API 兼容的结构化响应
3. 通过 TTLModelPool 管理模型生命周期，避免重复加载

整体调用链路：
  Router 层 → FunASREngine.transcribe() → TTLModelPool → FunASR AutoModel.generate()
                                        → _build_segments() → _build_speakers()
                                        → 返回 TranscriptionResult

本模块不依赖任何上游业务（如 minutes），可独立用于任何需要 STT 能力的场景。
"""

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
    """FunASR 库不可用时抛出的异常。

    当运行环境中未安装 funasr 包时，尝试加载模型会触发此异常。
    通常出现在：缺少 GPU 依赖、未安装完整依赖、或在纯 CPU 测试环境中运行。
    """

    pass


class FunASREngine:
    """FunASR 推理引擎，封装了模型加载、推理调用和结果转换的全部逻辑。

    该类是本服务的核心组件，职责包括：
    - 根据 model 参数（如 "cn_meeting"）查找对应的 ProfileSpec 配置
    - 通过 TTLModelPool 按需加载/复用 FunASR 模型实例
    - 调用模型进行推理，并将原始结果转为 OpenAI 兼容的 TranscriptionResult

    使用方式：
        engine = FunASREngine(settings=settings, model_pool=pool)
        result = engine.transcribe(audio_path, model="cn_meeting")

    属性:
        settings: 全局配置对象，包含模型缓存路径、设备类型等
        model_pool: 带 TTL 过期机制的模型对象池，线程安全
    """

    def __init__(self, *, settings: Settings, model_pool: TTLModelPool[Any]) -> None:
        """初始化推理引擎。

        参数:
            settings: 全局配置对象（Settings），包含 device、model_cache_dir 等配置项
            model_pool: 模型对象池（TTLModelPool），管理模型实例的缓存和生命周期。
                        泛型参数为 Any，因为 FunASR AutoModel 没有公开的类型定义。
        """
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
        """对指定音频文件执行语音转文字推理。

        这是引擎对外的主入口方法，完整流程：
        1. 根据 model 参数解析 ProfileSpec（模型组合配置）
        2. 从模型池获取或加载 FunASR 模型实例
        3. 构建推理参数并调用 FunASR generate()
        4. 解析原始结果，构建 segments（分句）和 speakers（说话人）
        5. 返回 OpenAI Whisper API 兼容的 TranscriptionResult

        参数:
            audio_path: 待转写的音频文件路径（通常是 Router 层写入的临时文件）
            model: 模型 profile 名称，对应 profiles.py 中定义的配置。
                   目前支持 "cn_meeting"（中文会议）和 "multilingual_rich"（多语种）。
                   默认 "cn_meeting"。
            language: 指定语言代码（如 "zh"、"en"），为 None 时使用 profile 的默认语言。
                      对于 multilingual_rich profile，默认为 "auto"（自动检测）。
            hotwords: 热词字符串（逗号分隔），用于提升特定词汇的识别准确率。
                      仅在 profile 支持热词时生效（如 cn_meeting 支持，multilingual_rich 不支持）。

        返回:
            TranscriptionResult: 包含完整转写文本、语言、时长、分句列表和说话人列表的结果对象。

        异常:
            RuntimeError: 当 FunASR 返回空结果时抛出
            FunASRUnavailableError: 当 FunASR 库未安装时抛出（由 _get_or_load_model 传播）
        """
        # 根据 model 名称（如 "cn_meeting"）查找对应的 ProfileSpec 配置
        profile = get_profile_spec(model)
        # 从模型池中获取已缓存的模型，或触发首次加载
        asr_model = self._get_or_load_model(profile.name.value)

        # 构建 FunASR generate() 方法所需的参数字典
        generate_kwargs: dict[str, Any] = {
            "input": str(audio_path),  # 音频文件路径（FunASR 接受字符串路径）
            "cache": {},  # 推理缓存，每次请求独立（空字典表示不复用）
            "language": language or profile.default_language,  # 语言设置，优先使用用户指定值
            "use_itn": True,  # 启用逆文本正则化（数字、日期等转为书面形式）
            "batch_size_s": 60,  # VAD 切分后每批最大秒数，控制 GPU 显存占用
            "merge_vad": True,  # 合并 VAD 切分的短片段，减少碎片化
            "merge_length_s": 15,  # 合并后每段最大秒数
        }
        # 仅当用户提供了热词且当前 profile 支持热词功能时才传入
        if hotwords and profile.supports_hotwords:
            generate_kwargs["hotword"] = hotwords

        logger.debug("Starting inference for profile={} file={}", profile.name.value, audio_path.name)
        t0 = time.monotonic()  # 使用单调时钟计时，不受系统时间调整影响

        # 调用 FunASR 模型执行推理，返回结果列表（通常只有一个元素）
        results = asr_model.generate(**generate_kwargs)
        if not results:
            raise RuntimeError("FunASR returned no transcription results.")

        # 提取第一个（也是唯一的）推理结果
        item = results[0]
        full_text = str(item.get("text", "")).strip()  # 完整转写文本
        sentence_info = item.get("sentence_info") or []  # 分句详情列表（包含时间戳、说话人等）
        lang = language or profile.default_language  # 最终使用的语言标识
        # 将 FunASR 原始分句数据转为 OpenAI 兼容的 Segment 列表
        segments = self._build_segments(sentence_info, full_text)
        # 从 segments 中汇总说话人统计信息
        speakers = self._build_speakers(segments)
        # 计算音频总时长（取所有 segment 的最大结束时间）
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

        # 组装并返回 OpenAI Whisper API 兼容的转写结果
        return TranscriptionResult(
            text=full_text,
            language=lang,
            duration=duration,
            segments=segments,
            x_speakers=speakers,  # x_ 前缀表示这是 OpenAI API 的扩展字段
        )

    def _get_or_load_model(self, cache_key: str):
        """从模型池中获取模型实例，若不存在或已过期则触发加载。

        该方法通过 TTLModelPool 实现了模型的懒加载和缓存复用：
        - 首次请求某个 profile 时，加载对应的 FunASR 模型（耗时较长，通常 10-30 秒）
        - 后续请求直接从池中获取已加载的模型实例（毫秒级）
        - 模型超过 TTL 未被使用时会自动过期，下次请求重新加载

        加载过程中会根据 ProfileSpec 配置组装多个子模型：
        - ASR 模型（必需）：核心语音识别模型
        - VAD 模型（必需）：语音活动检测，用于切分静音段
        - 标点模型（可选）：自动添加标点符号
        - 说话人模型（可选）：说话人分离（speaker diarization）

        参数:
            cache_key: 模型缓存键，即 profile 名称（如 "cn_meeting"）

        返回:
            FunASR AutoModel 实例，可直接调用 .generate() 进行推理

        异常:
            FunASRUnavailableError: 当 funasr 包未安装时抛出
        """

        def _loader():
            """模型加载闭包，仅在缓存未命中时由 TTLModelPool 调用。"""
            try:
                # 延迟导入 FunASR，因为它依赖 PyTorch 等大型库，
                # 在测试环境中可能不可用
                from funasr import AutoModel
            except ImportError as exc:
                raise FunASRUnavailableError(
                    "FunASR is not installed. Install the project dependencies to enable transcription."
                ) from exc

            logger.info("Loading model: cache_key={} device={}", cache_key, self.settings.device)
            t0 = time.monotonic()

            # 根据 cache_key 查找 ProfileSpec，获取各子模型的 model ID
            profile = get_profile_spec(cache_key)
            # 构建 AutoModel 初始化参数，每个子模型路径都经过 _resolve_model_path 解析
            model_kwargs: dict[str, Any] = {
                "model": self._resolve_model_path(profile.asr_model_id),  # ASR 主模型
                "vad_model": self._resolve_model_path(profile.vad_model_id),  # VAD 语音活动检测模型
                "device": self.settings.device,  # 推理设备（如 "cuda:0" 或 "cpu"）
                "trust_remote_code": True,  # 允许执行 ModelScope 模型仓库中的自定义代码
            }
            # 标点模型为可选（multilingual_rich profile 不使用标点模型）
            if profile.punc_model_id:
                model_kwargs["punc_model"] = self._resolve_model_path(profile.punc_model_id)
            # 说话人识别模型为可选
            if profile.speaker_model_id:
                model_kwargs["spk_model"] = self._resolve_model_path(profile.speaker_model_id)
            # 实例化 FunASR AutoModel，此步骤会加载模型权重到 GPU/CPU
            result = AutoModel(**model_kwargs)

            elapsed = time.monotonic() - t0
            logger.info("Model loaded: cache_key={} elapsed={:.3f}s", cache_key, elapsed)
            return result

        # 委托给模型池：命中缓存直接返回，未命中则调用 _loader 加载
        return self.model_pool.get_or_create(cache_key, _loader)

    def _resolve_model_path(self, model_id: str) -> str:
        """解析模型路径：优先使用本地缓存，无缓存时回退到 ModelScope 在线下载。

        本方法会在 model_cache_dir 下依次检查以下子目录：
        1. {cache_dir}/{model_id}           — 直接存放（download_models.py 默认方式）
        2. {cache_dir}/models/{model_id}    — ModelScope SDK 缓存结构
        3. {cache_dir}/hub/{model_id}       — 另一种常见缓存结构

        若找到本地目录则返回绝对路径字符串，FunASR 将直接从本地加载模型；
        若未找到则返回原始 model_id（如 "iic/speech_seaco_paraformer_large_asr_nat-..."），
        FunASR 会自动从 ModelScope 下载。

        参数:
            model_id: 模型标识符，格式为 "组织名/模型名"（如 "iic/SenseVoiceSmall"）

        返回:
            str: 本地模型目录的绝对路径，或原始 model_id（触发在线下载）
        """
        cache = self.settings.model_cache_dir.expanduser()  # 展开 ~ 为用户主目录
        # 依次尝试三种常见的本地缓存目录结构
        for prefix in ("", "models/", "hub/"):
            candidate = cache / prefix / model_id
            if candidate.is_dir():
                logger.debug("Resolved model path: {} -> {}", model_id, candidate)
                return str(candidate)
        # 本地无缓存，返回原始 ID，FunASR 将从 ModelScope 在线下载
        logger.debug("No local cache for model {}, will download from ModelScope", model_id)
        return model_id

    @staticmethod
    def _build_segments(sentence_info: list[dict[str, Any]], fallback_text: str) -> list[Segment]:
        """将 FunASR 原始分句信息转换为 OpenAI 兼容的 Segment 列表。

        FunASR 的 generate() 返回的 sentence_info 是一个字典列表，每个字典包含：
        - text: 该句文本
        - start / end: 起止时间（毫秒）
        - spk / speaker: 说话人标识（整数或字符串）
        - confidence: 置信度（浮点数）

        本方法将毫秒时间戳转换为秒（OpenAI Whisper API 约定），
        并统一说话人字段名称和类型。

        参数:
            sentence_info: FunASR 返回的原始分句信息列表。
                          每个元素是一个字典，包含 text、start、end、spk 等字段。
            fallback_text: 当 sentence_info 为空时的兜底文本（通常是完整转写文本）。
                          这种情况发生在模型未返回分句详情时（罕见但需处理）。

        返回:
            list[Segment]: OpenAI 兼容的分句列表，时间单位为秒。
        """
        # 当 FunASR 未返回分句信息时，构造一个包含完整文本的兜底 segment
        if not sentence_info:
            return [
                Segment(
                    id=0,
                    start=0.0,
                    end=1.0,  # 给一个默认的 1 秒时长，避免时长为 0
                    text=fallback_text,
                    x_speaker_id=None,
                    x_confidence=None,
                )
            ]

        segments: list[Segment] = []
        for index, item in enumerate(sentence_info):
            # FunASR 不同版本/模型使用不同的说话人字段名：'spk' 或 'speaker'
            speaker_raw = item.get("spk") or item.get("speaker")
            # 统一转为字符串类型（FunASR 可能返回整数如 0, 1, 2）
            speaker_id = str(speaker_raw) if speaker_raw is not None else None
            # FunASR 返回的时间戳单位是毫秒（整数）
            start_ms = int(item.get("start", 0))
            end_ms = int(item.get("end", start_ms + 1000))  # 无 end 时默认 1 秒时长
            segments.append(
                Segment(
                    id=index,
                    start=start_ms / 1000.0,  # 毫秒 → 秒，符合 OpenAI Whisper API 约定
                    end=end_ms / 1000.0,  # 毫秒 → 秒
                    text=str(item.get("text", "")).strip(),
                    x_speaker_id=speaker_id,  # x_ 前缀：OpenAI API 扩展字段
                    x_confidence=item.get("confidence"),  # x_ 前缀：OpenAI API 扩展字段
                )
            )
        return segments

    @staticmethod
    def _build_speakers(segments: list[Segment]) -> list[Speaker]:
        """从 Segment 列表中汇总说话人统计信息。

        遍历所有 segment，按说话人 ID 统计：
        - 累计发言时长（秒）
        - 发言片段数量

        最终按说话人 ID 排序后返回 Speaker 列表。

        参数:
            segments: 已构建的 Segment 列表（由 _build_segments 生成）

        返回:
            list[Speaker]: 说话人统计列表，按 speaker_id 升序排列。
                          若所有 segment 均无说话人信息，返回空列表。
        """
        totals: Counter[str] = Counter()  # 记录每个说话人的累计发言时长（秒）
        counts: Counter[str] = Counter()  # 记录每个说话人的发言片段数
        for seg in segments:
            if seg.x_speaker_id is None:
                continue  # 跳过没有说话人标识的片段
            # 累加该说话人的发言时长
            totals[seg.x_speaker_id] += seg.end - seg.start
            counts[seg.x_speaker_id] += 1
        # 按说话人 ID 排序，构建 Speaker 对象列表
        return [
            Speaker(
                speaker_id=sid,
                # 生成可读的显示名称：如 "spk_0" → "Spk 0"
                display_name=sid.replace("_", " ").title(),
                segment_count=counts[sid],
                total_duration=round(totals[sid], 3),  # 保留 3 位小数，避免浮点精度问题
            )
            for sid in sorted(counts)  # 按 speaker_id 字典序排列，保证输出稳定
        ]

    @staticmethod
    def _compute_duration(segments: list[Segment]) -> float:
        """计算音频总时长（秒）。

        取所有 segment 中最大的结束时间作为音频总时长。
        这是一种简化计算方式 — 假设最后一个 segment 的结束时间即为音频结尾。

        参数:
            segments: Segment 列表

        返回:
            float: 音频总时长（秒）。若 segments 为空，返回 0.0。
        """
        if not segments:
            return 0.0
        # 取所有片段中最晚的结束时间点作为总时长
        return max(seg.end for seg in segments)
