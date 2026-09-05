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
| M8 | Dockerfile + Hugging Face Space / model-repo deploy scripts | ✅ done |

Training all 15 categories and the deployed URL are added once the MVTec AD
download + training run completes; the pipeline and deploy path are in place.

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
   training  (local RTX 4060 · PyTorch + Anomalib · [train] extra)
   ────────────────────────────────────────────────────────────────
   MVTec AD (good only) ─► PatchCore / EfficientAD  ─► Engine.fit + test
                                                    ─► ONNX export  (Engine.export)
                                                    ─► artifacts/<model>/<category>/
                                                         model.onnx · preprocess.json · metrics.json
                                        │
                    publish ────────────┘  huggingface.co/bhadra244131/mddf-artifacts

   serving  (free CPU Hugging Face Docker Space · Torch-free · ~800 MB image)
   ────────────────────────────────────────────────────────────────
   image ─► FastAPI ─► preprocess ─► ONNXRuntime (CPU, 1 thread)
                    ─► anomaly map + score ─► threshold ─► verdict
                    ─► JET overlay · binary mask · connected-component boxes ─► base64 PNG
        React SPA ◄── JSON
```

### Design notes

- **No Grad-CAM.** Grad-CAM needs a trained classifier with class logits to backprop.
  PatchCore has neither; it already emits a pixel-level anomaly map from patch-to-
  memory-bank distances, and EfficientAD emits one from student–teacher disagreement.
  That map *is* the localization output — Grad-CAM would be a decorative wrapper around
  a signal we already have, so it was cut (and the unused EfficientNet with it).
- **Torch-free runtime.** Serving is ONNXRuntime + NumPy + OpenCV(headless); the
  deployed image ships no PyTorch or Anomalib (~800 MB vs. ~4 GB). Training deps are an
  opt-in `[train]` extra, and the ONNX export bakes resize + normalise + post-processing
  into the graph so preprocessing can't drift between train and serve.
- **Weights out of the image.** `model.onnx` + JSON per category live in a Hugging Face
  *model* repo and are pulled lazily on first request, then disk- and LRU-cached in RAM
  (`MDDF_REGISTRY_CACHE_SIZE`, default 4). Cold start touches one category, not fifteen.
- **Two models, one benchmark.** PatchCore for accuracy, EfficientAD for CPU latency;
  `mddf benchmark` reports measured image AUROC against the published PatchCore baseline
  plus pixel AUROC, AUPRO and p50/p95 CPU latency.

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

## Deploy

```bash
make train ARGS="--model all"     # train PatchCore + EfficientAD, 15 categories
make export                       # -> artifacts/<model>/<category>/model.onnx
make benchmark ARGS="--latency"   # -> artifacts/benchmark/{metrics,COMPARISON.md}
make publish-artifacts            # push ONNX + JSON to the HF model repo
make deploy-space                 # create/update the HF Docker Space (builds the image)
```

The Docker image (`Dockerfile`) is a two-stage build: Node builds `web/dist`, then a
`python:3.12-slim` stage installs only the runtime deps and runs
`mddf serve` on port 7860. It runs the same locally (`make docker`) and on the Space.

---

## Repository layout

| Path | What |
|---|---|
| `src/mddf/config.py`, `catalog.py` | settings + the 15-category catalogue (`mddf/resources/categories.yaml`) |
| `src/mddf/data/` | MVTec AD download/verify + Anomalib datamodule *(M1)* |
| `src/mddf/training/` | `train` / `evaluate` / `export` — needs `[train]` extra *(M2–M4)* |
| `src/mddf/inference/` | Torch-free registry, NumPy scoring, heatmaps *(M5)* |
| `src/mddf/api/` | FastAPI app, routes, schemas, structured errors, middleware |
| `src/mddf/benchmark/` | accuracy + latency reporting *(M3)* |
| `web/` | React + Vite + Tailwind SPA *(M7)* |
| `src/mddf/resources/` | `categories.yaml`, `patchcore.yaml`, `efficient_ad.yaml` |

## License

MIT.
