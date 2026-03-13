"""模型 Profile 配置模块。

本模块定义了 FunASR 引擎支持的模型组合方案（Profile），每个 Profile 对应一套
完整的 ASR 流水线配置，包括：语音识别模型、语音活动检测（VAD）模型、标点恢复模型、
说话人识别模型等。

**设计理念**：API 接口只暴露一个简单的 ``model`` 参数（如 ``"cn_meeting"``），
内部通过 Profile 映射到具体的多模型组合，对调用方屏蔽底层模型细节。

目前支持的 Profile：

- ``cn_meeting`` — 中文会议场景，使用 Paraformer-large 系列，支持热词和说话人识别。
- ``multilingual_rich`` — 多语种场景，使用 SenseVoice，自动检测语言，支持富文本标签。

新增 Profile 时，需要同步更新本文件的 ``PROFILE_SPECS`` 字典、
``scripts/download_models.py`` 以及 ``CLAUDE.md`` 中的文档。
"""

from dataclasses import dataclass
from enum import StrEnum


class ModelProfile(StrEnum):
    """模型 Profile 枚举。

    每个枚举值对应 API 请求中 ``model`` 参数可接受的值。
    继承自 ``StrEnum``，因此枚举值可以直接与字符串比较和序列化。

    Attributes:
        CN_MEETING: 中文会议 Profile，适用于中文会议转写场景。
        MULTILINGUAL_RICH: 多语种富文本 Profile，适用于多语种自动识别场景。
    """

    CN_MEETING = "cn_meeting"
    MULTILINGUAL_RICH = "multilingual_rich"


@dataclass(frozen=True, slots=True)
class ProfileSpec:
    """单个 Profile 的完整配置规格。

    使用 ``frozen=True`` 确保配置不可变（创建后不能修改字段值），
    ``slots=True`` 优化内存占用和属性访问速度。

    Attributes:
        name: 该 Profile 对应的枚举值，用于反向引用。
        display_name: 人类可读的显示名称，用于日志和调试信息。
        asr_model_id: 语音识别（ASR）模型的 ModelScope ID。
            这是核心模型，负责将语音转为文字。
        vad_model_id: 语音活动检测（VAD）模型的 ModelScope ID。
            用于检测音频中哪些片段包含语音，过滤静音和噪音。
        punc_model_id: 标点恢复模型的 ModelScope ID，可为 None。
            用于为识别结果自动添加标点符号。SenseVoice 等模型自带标点，
            此时设为 None 表示不需要额外的标点模型。
        speaker_model_id: 说话人识别模型的 ModelScope ID，可为 None。
            用于区分不同说话人（说话人日志化 / Speaker Diarization）。
            设为 None 表示该 Profile 不支持说话人识别。
        default_language: 默认语言代码。
            ``"zh"`` 表示中文，``"auto"`` 表示自动检测语言。
        supports_hotwords: 是否支持热词功能。
            热词可以提高特定词汇（如专业术语、人名）的识别准确率。
        enable_rich_tags: 是否启用富文本标签。
            SenseVoice 模型可输出情感、事件等标签，开启后会在结果中保留这些标签。
    """

    name: ModelProfile
    display_name: str
    asr_model_id: str
    vad_model_id: str
    punc_model_id: str | None
    speaker_model_id: str | None
    default_language: str
    supports_hotwords: bool
    enable_rich_tags: bool


# 全局 Profile 配置注册表
# key 为 ModelProfile 枚举值，value 为对应的完整配置规格
PROFILE_SPECS: dict[ModelProfile, ProfileSpec] = {
    # ---- 中文会议 Profile ----
    # 使用 Paraformer-large（高精度中文 ASR） + FSMN-VAD + CT-Punc + CAM++ 说话人识别
    # 适合：中文会议录音转写，支持热词增强
    ModelProfile.CN_MEETING: ProfileSpec(
        name=ModelProfile.CN_MEETING,
        display_name="Chinese Meeting",
        # Paraformer-large: 阿里达摩院高精度中文语音识别模型，支持 SEAco 热词增强
        asr_model_id="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        # FSMN-VAD: 基于前馈序列记忆网络的语音活动检测，轻量高效
        vad_model_id="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        # CT-Transformer: 基于 Transformer 的中英文标点恢复模型
        punc_model_id="iic/punc_ct-transformer_cn-en-common-vocab471067-large",
        # CAM++: 基于 CAM++ 架构的说话人嵌入模型，用于说话人日志化
        speaker_model_id="iic/speech_campplus_sv_zh-cn_16k-common",
        default_language="zh",  # 默认中文
        supports_hotwords=True,  # Paraformer SEAco 支持热词
        enable_rich_tags=False,  # Paraformer 不输出富文本标签
    ),
    # ---- 多语种富文本 Profile ----
    # 使用 SenseVoice（多语种 ASR） + FSMN-VAD + CAM++ 说话人识别
    # 适合：多语种场景，自动语言检测，附带情感/事件标签
    ModelProfile.MULTILINGUAL_RICH: ProfileSpec(
        name=ModelProfile.MULTILINGUAL_RICH,
        display_name="Multilingual Rich",
        # SenseVoice Small: 支持中/英/日/韩/粤等多语种，内置标点和富文本标签
        asr_model_id="iic/SenseVoiceSmall",
        # 复用同一个 FSMN-VAD 模型
        vad_model_id="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        # SenseVoice 自带标点输出，不需要额外标点模型
        punc_model_id=None,
        # 复用 CAM++ 说话人模型
        speaker_model_id="iic/speech_campplus_sv_zh-cn_16k-common",
        default_language="auto",  # 自动检测语言
        supports_hotwords=False,  # SenseVoice 不支持热词
        enable_rich_tags=True,  # 启用情感、事件等富文本标签
    ),
}


def resolve_profile(value: str | ModelProfile | None) -> ModelProfile:
    """将用户输入的 model 参数解析为标准的 ModelProfile 枚举值。

    支持三种输入形式：
    - ``None`` → 返回默认 Profile（``CN_MEETING``）
    - ``ModelProfile`` 枚举值 → 原样返回
    - 字符串（如 ``"cn_meeting"``） → 转换为对应的枚举值

    Args:
        value: 用户传入的 model 参数，可以是字符串、枚举值或 None。

    Returns:
        解析后的 ModelProfile 枚举值。

    Raises:
        ValueError: 当传入的字符串不是合法的 Profile 名称时抛出。
    """
    if value is None:
        return ModelProfile.CN_MEETING  # 未指定时默认使用中文会议 Profile
    if isinstance(value, ModelProfile):
        return value  # 已经是枚举值，无需转换
    # 利用 StrEnum 的构造函数将字符串转为枚举；无效值会抛出 ValueError
    return ModelProfile(value)


def get_profile_spec(value: str | ModelProfile | None) -> ProfileSpec:
    """根据 model 参数获取完整的 Profile 配置规格。

    这是外部模块获取 Profile 配置的主入口函数，内部先调用 ``resolve_profile``
    解析参数，再从 ``PROFILE_SPECS`` 字典中查找对应配置。

    Args:
        value: 用户传入的 model 参数，可以是字符串、枚举值或 None。

    Returns:
        对应的 ProfileSpec 配置对象，包含该 Profile 的全部模型 ID 和特性开关。

    Raises:
        ValueError: 当 model 参数不是合法的 Profile 名称时抛出。
        KeyError: 当解析出的 Profile 未在 PROFILE_SPECS 中注册时抛出（正常不应发生）。
    """
    return PROFILE_SPECS[resolve_profile(value)]
