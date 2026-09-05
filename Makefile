.DEFAULT_GOAL := help
PY ?= python

.PHONY: help install install-train lint typecheck test check serve data train export benchmark web docker

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install runtime + dev deps (Torch-free)
	$(PY) -m pip install -e ".[dev]"

install-train: ## Install everything incl. Torch/Anomalib (for local GPU training)
	$(PY) -m pip install -e ".[train,dev]"

lint: ## ruff
	ruff check src tests
	ruff format --check src tests

typecheck: ## mypy
	mypy

test: ## pytest
	pytest

check: lint typecheck test ## Everything CI runs

serve: ## Run the API locally with reload
	$(PY) -m mddf.cli serve --reload --host 127.0.0.1

data: ## Download + verify MVTec AD (M1)
	$(PY) -m mddf.cli data

train: ## Train models (M2/M3), e.g. make train ARGS="--model patchcore --category leather"
	$(PY) -m mddf.cli train $(ARGS)

export: ## Export ONNX artifacts + split backbone/memory bank (M4)
	$(PY) -m mddf.cli export $(ARGS)

benchmark: ## Accuracy + latency table (M3)
	$(PY) -m mddf.cli benchmark $(ARGS)

web: ## Build the React SPA into web/dist (M7)
	cd web && npm ci && npm run build

docker: ## Build the deployment image
	docker build -t mddf:local .
