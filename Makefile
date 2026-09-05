.DEFAULT_GOAL := help
PY ?= python

.PHONY: help install install-train install-cuda lint typecheck test check serve data train export benchmark report verify web docker publish-artifacts deploy-space

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install runtime + dev deps (Torch-free)
	$(PY) -m pip install -e ".[dev]"

install-train: ## Install everything incl. Torch/Anomalib (for local GPU training)
	$(PY) -m pip install -e ".[train,dev]"
	@echo "PyPI ships CPU-only torch. For NVIDIA GPUs, overlay the CUDA build:"
	@echo "  $(PY) -m pip install --index-url https://download.pytorch.org/whl/cu126 \\"
	@echo "        torch==2.14.0+cu126 torchvision==0.29.0+cu126"

install-cuda: ## Overlay the CUDA 12.6 torch build (run after install-train, needs an NVIDIA GPU)
	$(PY) -m pip install --index-url https://download.pytorch.org/whl/cu126 \
		"torch==2.14.0+cu126" "torchvision==0.29.0+cu126"

lint: ## ruff
	ruff check src tests deploy
	ruff format --check src tests deploy

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

benchmark: ## Accuracy + latency table -> docs/RESULTS.md
	$(PY) -m mddf.cli benchmark $(ARGS)

report: ## Operating-point analysis (recall vs false-alarm) from the ONNX models
	$(PY) -m mddf.cli report $(ARGS)

verify: ## Parity gate: INT8 export must match fp32 image AUROC within tolerance
	$(PY) -m mddf.cli verify $(ARGS)

web: ## Build the React SPA into web/dist (M7)
	cd web && npm ci && npm run build

docker: ## Build the deployment image
	docker build -t mddf:local .

publish-artifacts: ## Upload exported ONNX + metrics to the HF model repo
	$(PY) deploy/publish_artifacts.py $(ARGS)

deploy-space: ## Create/update the HF Docker Space and push the app
	$(PY) deploy/deploy_space.py $(ARGS)
