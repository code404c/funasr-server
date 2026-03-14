"""SRT / VTT / TXT formatters for transcription results.

本模块负责将引擎产出的 TranscriptionResult 转换为非 JSON 的纯文本格式。
当调用方指定 response_format 为 "text"、"srt" 或 "vtt" 时，
router 层会调用本模块的对应函数进行格式化。

支持的输出格式:
  - text (纯文本): 直接返回转写文本，无时间戳
  - srt (SubRip 字幕): 编号 + 时间戳 + 文本，广泛用于视频字幕
  - vtt (WebVTT 字幕): Web 标准字幕格式，浏览器原生支持

SRT 与 VTT 的主要区别:
  - SRT 时间戳使用逗号分隔毫秒（00:01:30,500）
  - VTT 时间戳使用点号分隔毫秒（00:01:30.500）
  - VTT 文件必须以 "WEBVTT" 头部开始
  - SRT 每个字幕块需要有序号编号
"""

from __future__ import annotations

from funasr_server.schemas import TranscriptionResult


def format_txt(result: TranscriptionResult) -> str:
    """将转写结果格式化为纯文本。

    最简单的输出格式，直接返回完整的转写文本，
    不包含时间戳、说话人等元信息。

    Args:
        result: 引擎产出的转写结果对象

    Returns:
        纯文本字符串
    """
    return result.text


def _format_timestamp(seconds: float, *, vtt: bool = False) -> str:
    """将秒数转换为字幕格式的时间戳字符串。

    时间戳格式: HH:MM:SS,mmm (SRT) 或 HH:MM:SS.mmm (VTT)

    转换流程:
      秒 → 总毫秒 → 拆分为 时/分/秒/毫秒 → 格式化为字符串

    Args:
        seconds: 时间点（单位: 秒），如 90.5 表示 1 分 30 秒 500 毫秒
        vtt: 是否使用 VTT 格式。True 时毫秒分隔符为 "."，False 时为 ","

    Returns:
        格式化后的时间戳字符串，如 "00:01:30,500" 或 "00:01:30.500"
    """
    # 先将秒转为总毫秒数（四舍五入），避免浮点精度问题
    total_ms = round(seconds * 1000)

    # 依次拆分出 时、分、秒、毫秒（使用 divmod 整除取余）
    hours, remainder = divmod(total_ms, 3_600_000)  # 1 小时 = 3,600,000 毫秒
    minutes, remainder = divmod(remainder, 60_000)  # 1 分钟 = 60,000 毫秒
    secs, millis = divmod(remainder, 1_000)  # 1 秒 = 1,000 毫秒

    # SRT 用逗号分隔毫秒，VTT 用点号分隔毫秒
    separator = "." if vtt else ","

    # 各字段补零到固定宽度: 时/分/秒各 2 位，毫秒 3 位
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def format_srt(result: TranscriptionResult) -> str:
    """将转写结果格式化为 SRT 字幕格式。

    SRT (SubRip Text) 是最常见的字幕格式，被大多数视频播放器支持。
    每个字幕块由三部分组成:
      1. 序号（从 1 开始递增）
      2. 时间轴（起始时间 --> 结束时间）
      3. 字幕文本

    字幕块之间用空行分隔。

    输出示例:
      1
      00:00:00,000 --> 00:00:03,500
      你好，欢迎参加会议。

      2
      00:00:04,200 --> 00:00:08,100
      今天我们讨论第一季度的业绩。

    Args:
        result: 引擎产出的转写结果对象

    Returns:
        SRT 格式的字幕字符串
    """
    lines: list[str] = []

    # 遍历每个语音片段，start=1 表示序号从 1 开始（SRT 规范要求）
    for index, segment in enumerate(result.segments, start=1):
        lines.extend(
            [
                str(index),  # 第 1 行: 字幕块序号
                # 第 2 行: 时间轴，格式为 "起始时间 --> 结束时间"
                f"{_format_timestamp(segment.start)} --> {_format_timestamp(segment.end)}",
                segment.text,  # 第 3 行: 字幕文本内容
                "",  # 空行作为字幕块之间的分隔符
            ]
        )

    # 用换行符拼接所有行，strip() 去掉末尾多余的空行
    return "\n".join(lines).strip()


def format_vtt(result: TranscriptionResult) -> str:
    """将转写结果格式化为 WebVTT 字幕格式。

    WebVTT (Web Video Text Tracks) 是 W3C 标准的字幕格式，
    浏览器 <track> 元素原生支持，适合 Web 端视频播放。

    与 SRT 的区别:
      - 文件必须以 "WEBVTT" 开头
      - 时间戳用点号（.）分隔毫秒，而非逗号
      - 字幕块不需要序号（但加了也合法）

    输出示例:
      WEBVTT

      00:00:00.000 --> 00:00:03.500
      你好，欢迎参加会议。

      00:00:04.200 --> 00:00:08.100
      今天我们讨论第一季度的业绩。

    Args:
        result: 引擎产出的转写结果对象

    Returns:
        WebVTT 格式的字幕字符串
    """
    # VTT 文件必须以 "WEBVTT" 头部开始，后跟一个空行
    lines = ["WEBVTT", ""]

    for segment in result.segments:
        lines.extend(
            [
                # 时间轴行，vtt=True 使用点号分隔毫秒
                f"{_format_timestamp(segment.start, vtt=True)} --> {_format_timestamp(segment.end, vtt=True)}",
                segment.text,  # 字幕文本
                "",  # 空行作为字幕块分隔符
            ]
        )

    # 用换行符拼接，strip() 去掉末尾多余的空行
    return "\n".join(lines).strip()
