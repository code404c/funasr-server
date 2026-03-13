FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml /app/pyproject.toml
COPY README.md /app/README.md
COPY src /app/src
COPY scripts /app/scripts

RUN python -m pip install --upgrade pip && python -m pip install .

EXPOSE 8000
CMD ["uvicorn", "funasr_server.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
