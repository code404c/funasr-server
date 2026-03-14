"""模型列表路由 — 提供 OpenAI 兼容的 GET /v1/models 端点。

返回所有可用的模型 profile，格式与 OpenAI ``GET /v1/models`` 一致。
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from funasr_server.profiles import PROFILE_SPECS

router = APIRouter(prefix="/v1", tags=["models"])


class ModelObject(BaseModel):
    """单个模型的描述信息（OpenAI 兼容格式）。"""

    id: str
    object: str = "model"
    owned_by: str = "funasr-server"


class ModelListResponse(BaseModel):
    """模型列表响应（OpenAI 兼容格式）。"""

    object: str = "list"
    data: list[ModelObject]


@router.get("/models")
def list_models() -> ModelListResponse:
    """返回所有可用模型的列表。"""
    data = [ModelObject(id=spec.name.value) for spec in PROFILE_SPECS.values()]
    return ModelListResponse(data=data)
