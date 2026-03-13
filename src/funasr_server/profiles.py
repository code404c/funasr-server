from dataclasses import dataclass
from enum import StrEnum


class ModelProfile(StrEnum):
    CN_MEETING = "cn_meeting"
    MULTILINGUAL_RICH = "multilingual_rich"


@dataclass(frozen=True, slots=True)
class ProfileSpec:
    name: ModelProfile
    display_name: str
    asr_model_id: str
    vad_model_id: str
    punc_model_id: str | None
    speaker_model_id: str | None
    default_language: str
    supports_hotwords: bool
    enable_rich_tags: bool


PROFILE_SPECS: dict[ModelProfile, ProfileSpec] = {
    ModelProfile.CN_MEETING: ProfileSpec(
        name=ModelProfile.CN_MEETING,
        display_name="Chinese Meeting",
        asr_model_id="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        vad_model_id="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        punc_model_id="iic/punc_ct-transformer_cn-en-common-vocab471067-large",
        speaker_model_id="iic/speech_campplus_sv_zh-cn_16k-common",
        default_language="zh",
        supports_hotwords=True,
        enable_rich_tags=False,
    ),
    ModelProfile.MULTILINGUAL_RICH: ProfileSpec(
        name=ModelProfile.MULTILINGUAL_RICH,
        display_name="Multilingual Rich",
        asr_model_id="iic/SenseVoiceSmall",
        vad_model_id="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        punc_model_id=None,
        speaker_model_id="iic/speech_campplus_sv_zh-cn_16k-common",
        default_language="auto",
        supports_hotwords=False,
        enable_rich_tags=True,
    ),
}


def resolve_profile(value: str | ModelProfile | None) -> ModelProfile:
    if value is None:
        return ModelProfile.CN_MEETING
    if isinstance(value, ModelProfile):
        return value
    return ModelProfile(value)


def get_profile_spec(value: str | ModelProfile | None) -> ProfileSpec:
    return PROFILE_SPECS[resolve_profile(value)]
