"""结构化错误响应模块。

定义转写服务的自定义异常类型，由全局异常处理器捕获后返回统一格式的 JSON 响应。
响应格式遵循 OpenAI 风格::

    {"error": {"message": "...", "type": "...", "error_id": "..."}}
"""

from __future__ import annotations

import uuid


class TranscriptionError(Exception):
    """转写过程中的结构化错误。

    Attributes:
        error_id: UUID 错误标识，用于日志追踪和问题定位。
        message: 面向调用方的错误描述。
        status_code: HTTP 状态码。
        error_type: 错误类型标识（如 "invalid_request_error", "service_unavailable"）。
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 500,
        error_type: str = "transcription_error",
    ) -> None:
        super().__init__(message)
        self.error_id = str(uuid.uuid4())
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
