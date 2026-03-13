ARG BASE_IMAGE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

# 系统依赖：curl（健康检查）、ffmpeg（音频格式转换）
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates curl ffmpeg && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 创建非 root 用户（安全最佳实践）
RUN useradd --create-home --shell /bin/bash --uid 1000 appuser || true

# 预创建模型缓存目录并设置权限（在切换用户之前，以 root 身份执行）
RUN mkdir -p /modelscope-cache && chown 1000:1000 /modelscope-cache

USER appuser
ENV HOME=/home/appuser PATH=/home/appuser/.local/bin:$PATH
WORKDIR $HOME/app

# 安装 uv（从官方镜像复制，uv 会自动安装 pyproject.toml 要求的 Python 版本）
COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /bin/uv

# 先装依赖（利用 layer cache，pyproject.toml/uv.lock 不变时跳过）
RUN --mount=type=cache,target=/home/appuser/.cache/uv,uid=1000,gid=1000 \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=README.md,target=README.md \
    uv sync --frozen --compile-bytecode --no-install-project --no-dev

# 复制源码并安装项目本身
COPY --chown=appuser . .
RUN --mount=type=cache,target=/home/appuser/.cache/uv,uid=1000,gid=1000 \
    uv sync --frozen --compile-bytecode --no-dev

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 DO_NOT_TRACK=1
ENV PATH="$HOME/app/.venv/bin:$PATH"

EXPOSE 8000
CMD ["uvicorn", "funasr_server.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
