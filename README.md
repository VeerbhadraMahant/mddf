# Manufacturing Defect Detection System (MDDF)

Unsupervised visual quality inspection on the [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
dataset. Models are trained **only on defect-free images** — the realistic setting for a
factory line, where defects are rare and their appearance is open-ended — and served
**CPU-only** behind a FastAPI service with a React inspection UI.

Two methods are trained per category and benchmarked head-to-head:

| Model | Role | Backbone | Idea |
|---|---|---|---|
| **PatchCore** | accuracy | WideResNet-50 | coreset memory bank of normal patch embeddings; anomaly = distance to nearest normal patch |
| **EfficientAD** | latency | custom PDN | student–teacher + autoencoder distillation; designed for real-time edge inference |

Both produce a **native pixel-level anomaly map**, upsampled and overlaid on the input
for operator-facing localization — no Grad-CAM (see [Design notes](#design-notes)).

---

## Status

| Milestone | Scope | State |
|---|---|---|
| M0 | Scaffold, config, CI, FastAPI skeleton | ✅ done |
| M1 | MVTec AD download + verify, Anomalib datamodule | ✅ done |
| M2 | PatchCore / EfficientAD training + eval pipeline | ✅ done |
| M3 | Benchmark aggregation (accuracy vs baselines + CPU latency) | ✅ done |
| M4 | ONNX export for Torch-free serving | ✅ done |
| M5 | Torch-free inference core (ONNXRuntime + NumPy) | ✅ done |
| M6 | FastAPI `predict` / `predict/batch` endpoints | ✅ done |
| M7 | React SPA (Vite + Tailwind) | ✅ done |
| M8 | Docker + Hugging Face Space deploy | ⬜ |

Training runs (all 15 categories) and the deployed URL are filled in as M7/M8 land.

Results table (image AUROC vs. published PatchCore baseline) lands at M3.

---

## Quickstart

```bash
python -m pip install -e ".[dev]"     # runtime + dev tooling (no PyTorch)
make check                            # ruff + mypy + pytest
make serve                            # http://127.0.0.1:7860/api/docs
```

Local training needs an NVIDIA GPU and the heavier stack:

```bash
python -m pip install -e ".[train,dev]"                  # + PyTorch, Anomalib, Lightning
make install-cuda                                         # overlay the CUDA 12.6 torch build
make data                                                 # download + verify MVTec AD  (M1)
make train ARGS="--model patchcore --category leather"    # (M2)
```

> PyPI's `torch` is CPU-only; `make install-cuda` replaces it with `torch==2.14.0+cu126`
> from the PyTorch index. Verified against an RTX 4060 (8 GB, Ada / sm_89).

---

## Architecture

```
                 ┌── training (local RTX 4060, PyTorch + Anomalib) ──┐
 MVTec AD  ─────► │  PatchCore / EfficientAD  ──►  ONNX export        │
 (good only)      │                               ├─ backbone.onnx (shared, ~100 MB)
                 │                               └─ <category>/memory_bank.npy (~6 MB)
                 └──────────────────────────┬───────────────────────┘
                                            ▼   published to a Hugging Face model repo
             ┌── serving (free CPU HF Space, Torch-free) ──────────────┐
 image  ───► │ FastAPI ─► preprocess ─► ONNXRuntime backbone           │
             │           ─► NumPy nearest-neighbour scoring            │
             │           ─► anomaly map ─► heatmap / mask / bboxes     │
             │  React SPA  ◄── JSON + base64 PNG overlays              │
             └───────────────────────────────────────────────────────┘
```

### Design notes

- **No Grad-CAM.** Grad-CAM needs a trained classifier with class logits to backprop.
  PatchCore has neither; it already yields a pixel-level anomaly map from patch-to-
  memory-bank distances. That map *is* the localization output. Grad-CAM here would be
  a decorative wrapper around a signal we already have — so it was cut, and EfficientNet
  with it.
- **Decoupled artifacts.** Exporting each category as a self-contained ONNX bakes the
  270 MB backbone in 15× → ~4 GB, which will not fit a free Space. Instead: one shared
  `backbone.onnx` + 15 small `memory_bank.npy` files, with nearest-neighbour scoring
  done in NumPy at request time. Total ≈ 190 MB. A parity test (M4) asserts the split
  reproduces Anomalib's own AUROC within 1e-3.
- **Torch-free runtime.** Inference is ONNXRuntime + NumPy, so the deployed image ships
  no PyTorch (~800 MB vs. ~4 GB). Training dependencies are an opt-in `[train]` extra.
- **Lazy, LRU-cached model loading.** Artifacts are pulled per-category from the HF
  model repo on first use and cached on disk + in RAM, so cold start touches one
  category, not fifteen.

---

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/v1/health`, `/api/v1/ready` | liveness / readiness |
| `GET` | `/api/v1/categories` | 15 categories, available models, measured + published metrics |
| `GET` | `/api/v1/benchmark` | accuracy + CPU-latency comparison table |
| `POST` | `/api/v1/predict` | `multipart`: image + `category` + `model` → verdict, score, heatmap |
| `POST` | `/api/v1/predict/batch` | multi-image variant |
| `GET` | `/metrics` | Prometheus |
| `GET` | `/api/docs` | OpenAPI / Swagger UI |

Errors are `application/problem+json` (RFC 9457) with a stable `type` slug and the
request id — never a stack trace or HTML page.

---

## Repository layout

| Path | What |
|---|---|
| `src/mddf/config.py`, `catalog.py` | settings + the 15-category catalogue (`configs/categories.yaml`) |
| `src/mddf/data/` | MVTec AD download/verify + Anomalib datamodule *(M1)* |
| `src/mddf/training/` | `train` / `evaluate` / `export` — needs `[train]` extra *(M2–M4)* |
| `src/mddf/inference/` | Torch-free registry, NumPy scoring, heatmaps *(M5)* |
| `src/mddf/api/` | FastAPI app, routes, schemas, structured errors, middleware |
| `src/mddf/benchmark/` | accuracy + latency reporting *(M3)* |
| `web/` | React + Vite + Tailwind SPA *(M7)* |
| `configs/` | `categories.yaml`, `patchcore.yaml`, `efficient_ad.yaml` |

## License

MIT.
