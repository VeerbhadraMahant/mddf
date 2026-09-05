"""On-disk layout for trained models and their metrics.

```
artifacts/
├── patchcore/<category>/
│   ├── weights/model.ckpt          # Lightning checkpoint (M2)
│   ├── model.onnx                  # exported graph            (M4)
│   ├── preprocess.json             # PreprocessSpec            (M4)
│   ├── metrics.json                # image/pixel AUROC, AUPRO, F1, thresholds
│   └── run.json                    # config + env provenance
├── efficient_ad/<category>/ ...
└── benchmark/
    ├── metrics.json                # aggregate the API reads   (M3)
    └── latency.json                # CPU p50/p95               (M3)
```
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mddf.config import ModelName, get_settings


def model_root(model: ModelName, *, root: Path | None = None) -> Path:
    base = root or get_settings().artifacts_dir
    return base / model


def category_dir(model: ModelName, category: str, *, root: Path | None = None) -> Path:
    return model_root(model, root=root) / category


def checkpoint_path(model: ModelName, category: str, *, root: Path | None = None) -> Path:
    return category_dir(model, category, root=root) / "weights" / "model.ckpt"


def onnx_path(model: ModelName, category: str, *, root: Path | None = None) -> Path:
    return category_dir(model, category, root=root) / "model.onnx"


def preprocess_path(model: ModelName, category: str, *, root: Path | None = None) -> Path:
    return category_dir(model, category, root=root) / "preprocess.json"


def metrics_path(model: ModelName, category: str, *, root: Path | None = None) -> Path:
    return category_dir(model, category, root=root) / "metrics.json"


def run_path(model: ModelName, category: str, *, root: Path | None = None) -> Path:
    return category_dir(model, category, root=root) / "run.json"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


__all__ = [
    "category_dir",
    "checkpoint_path",
    "metrics_path",
    "model_root",
    "onnx_path",
    "preprocess_path",
    "read_json",
    "run_path",
    "write_json",
]
