---
title: Manufacturing Defect Detection
emoji: 🔎
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Unsupervised visual defect detection (PatchCore + EfficientAD), CPU-only.
---

# Manufacturing Defect Detection

Unsupervised visual quality inspection on the MVTec AD dataset. Two models per
category (PatchCore for accuracy, EfficientAD for latency) trained on defect-free
images only, exported to ONNX and served **CPU-only** behind FastAPI with a React
inspection UI.

- `/` — React SPA (upload an image, get a verdict + anomaly heatmap + boxes).
- `/api/docs` — OpenAPI.
- `/api/v1/benchmark` — measured image AUROC vs. the published PatchCore baseline.

Model artifacts are pulled lazily at runtime from the
[`bhadra244131/mddf-artifacts`](https://huggingface.co/bhadra244131/mddf-artifacts)
model repo, so the image itself ships no weights and no PyTorch.

Source: https://github.com/VeerbhadraMahant/mddf
