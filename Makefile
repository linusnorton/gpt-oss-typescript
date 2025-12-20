.PHONY: help install install-dev install-vllm install-all setup system-info \
        lint format typecheck test test-cov clean \
        serve-vllm eval train-dapt train-sft \
        collect-repos build-corpus \
        docker-build docker-build-train docker-build-eval docker-build-vllm \
        docker-up docker-down

# Default target
help:
	@echo "Nemotron TypeScript Post-Training Harness"
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install core dependencies with uv"
	@echo "  make install-dev    Install with dev dependencies"
	@echo "  make install-vllm   Install with vLLM dependencies"
	@echo "  make install-all    Install all dependencies"
	@echo "  make setup          Full environment setup (venv + deps + validation)"
	@echo "  make system-info    Collect system information (GPU, CUDA, etc.)"
	@echo ""
	@echo "Development:"
	@echo "  make lint           Run linter (ruff)"
	@echo "  make format         Format code (ruff)"
	@echo "  make typecheck      Run type checker (mypy)"
	@echo "  make test           Run tests"
	@echo "  make test-cov       Run tests with coverage"
	@echo "  make clean          Clean build artifacts"
	@echo ""
	@echo "Data Pipeline:"
	@echo "  make collect-repos  Collect GitHub repositories"
	@echo "  make build-corpus   Build DAPT corpus from collected repos"
	@echo ""
	@echo "Training:"
	@echo "  make train-dapt     Run Domain Adaptive Pre-Training"
	@echo "  make train-sft      Run Supervised Fine-Tuning"
	@echo ""
	@echo "Inference & Evaluation:"
	@echo "  make serve-vllm     Start vLLM server"
	@echo "  make eval           Run NeMo evaluation"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build        Build all Docker images"
	@echo "  make docker-build-train  Build training image"
	@echo "  make docker-build-eval   Build evaluation image"
	@echo "  make docker-build-vllm   Build vLLM serving image"
	@echo "  make docker-up           Start all services"
	@echo "  make docker-down         Stop all services"

# Environment variables
PYTHON ?= python3.11
UV ?= uv
VENV_DIR ?= .venv
DATA_DIR ?= ./data
OUTPUT_DIR ?= ./outputs
MODEL_ID ?= nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16

# Check if .env exists, if not copy from example
.env:
	@if [ ! -f .env ]; then \
		echo "Creating .env from .env.example..."; \
		cp .env.example .env; \
		echo "Please edit .env with your credentials"; \
	fi

# Installation targets
install: .env
	$(UV) venv $(VENV_DIR) --python $(PYTHON)
	$(UV) pip install -e . --python $(VENV_DIR)/bin/python

install-dev: .env
	$(UV) venv $(VENV_DIR) --python $(PYTHON)
	$(UV) pip install -e ".[dev]" --python $(VENV_DIR)/bin/python
	$(VENV_DIR)/bin/pre-commit install

install-vllm: .env
	$(UV) venv $(VENV_DIR) --python $(PYTHON)
	$(UV) pip install -e ".[vllm]" --python $(VENV_DIR)/bin/python

install-all: .env
	$(UV) venv $(VENV_DIR) --python $(PYTHON)
	$(UV) pip install -e ".[dev,vllm,nemo]" --python $(VENV_DIR)/bin/python
	$(VENV_DIR)/bin/pre-commit install

setup: .env
	@bash scripts/setup_env.sh

system-info:
	@bash scripts/system_info.sh

# Development targets
lint:
	$(VENV_DIR)/bin/ruff check src tests

format:
	$(VENV_DIR)/bin/ruff format src tests
	$(VENV_DIR)/bin/ruff check --fix src tests

typecheck:
	$(VENV_DIR)/bin/mypy src

test:
	$(VENV_DIR)/bin/pytest tests/

test-cov:
	$(VENV_DIR)/bin/pytest tests/ --cov=src --cov-report=html --cov-report=term

clean:
	rm -rf build dist *.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf htmlcov .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Data pipeline targets
collect-repos:
	$(VENV_DIR)/bin/python scripts/collect_github_repos.py \
		--config configs/github_collection.yaml \
		--output $(DATA_DIR)/repos

build-corpus:
	$(VENV_DIR)/bin/python scripts/build_dapt_corpus.py \
		--config configs/dapt_train.yaml \
		--input $(DATA_DIR)/repos \
		--output $(DATA_DIR)/corpus

# Training targets
train-dapt:
	$(VENV_DIR)/bin/python scripts/run_dapt_train.py \
		--config configs/dapt_train.yaml

train-sft:
	$(VENV_DIR)/bin/python scripts/apply_agentic_sft_dataset.py \
		--config configs/sft_data.yaml

# Inference & Evaluation targets
serve-vllm:
	@bash scripts/serve_vllm.sh --model $(MODEL_ID)

eval:
	@bash scripts/run_nemo_eval.sh

# Docker targets
DOCKER_COMPOSE = docker compose -f docker/docker-compose.yml
CUDA_VERSION ?= 12.4.1

docker-build: docker-build-train docker-build-eval docker-build-vllm

docker-build-train:
	docker build \
		--build-arg CUDA_VERSION=$(CUDA_VERSION) \
		-f docker/Dockerfile.train \
		-t nemotron-ts-train:latest .

docker-build-eval:
	docker build \
		--build-arg CUDA_VERSION=$(CUDA_VERSION) \
		-f docker/Dockerfile.eval \
		-t nemotron-ts-eval:latest .

docker-build-vllm:
	docker build \
		--build-arg CUDA_VERSION=$(CUDA_VERSION) \
		-f docker/Dockerfile.vllm \
		-t nemotron-ts-vllm:latest .

docker-up:
	$(DOCKER_COMPOSE) up -d

docker-down:
	$(DOCKER_COMPOSE) down

# Create directories if they don't exist
$(DATA_DIR):
	mkdir -p $(DATA_DIR)

$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)
