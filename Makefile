UV := uv
PYTHON := $(UV) run python
PYTEST := $(UV) run pytest
RUFF := $(UV) run ruff
DOCKER_COMPOSE := docker compose

.PHONY: help install dev test format lint check download-models docker-up docker-down

help:
	@echo "开发:"
	@echo "  make install          - 同步依赖 (uv sync --extra dev)"
	@echo "  make dev              - 启动开发服务器 (热重载)"
	@echo ""
	@echo "代码质量:"
	@echo "  make format           - ruff format"
	@echo "  make lint             - ruff check --fix"
	@echo "  make test             - pytest"
	@echo "  make check            - format-check + lint + test"
	@echo ""
	@echo "模型:"
	@echo "  make download-models  - 预下载 FunASR 模型"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-up        - docker compose up --build"
	@echo "  make docker-down      - docker compose down"

install:
	$(UV) sync --extra dev

dev:
	$(UV) run uvicorn funasr_server.app:create_app --factory --reload --host 0.0.0.0 --port 8000

format:
	$(RUFF) format src tests scripts

lint:
	$(RUFF) check --fix src tests scripts

test:
	$(PYTEST) -v

check:
	$(RUFF) format --check src tests scripts
	$(RUFF) check src tests scripts
	$(PYTEST) -v

download-models:
	$(PYTHON) scripts/download_models.py

docker-up:
	$(DOCKER_COMPOSE) up --build

docker-down:
	$(DOCKER_COMPOSE) down
