"""Typed loaders for ``mddf/resources/patchcore.yaml`` and ``efficient_ad.yaml``.

Shared by the data pipeline (image / batch sizes) and by training / export
(M2-M4). Kept deliberately small — only the fields the pipeline actually reads are
modelled; anything else in the YAML is ignored.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import yaml
from pydantic import BaseModel, Field

from mddf.config import CONFIGS_DIR, ModelName


class DataCfg(BaseModel):
    image_size: int = 256
    center_crop: int | None = None
    train_batch_size: int = 8
    eval_batch_size: int = 8
    num_workers: int = 4


class ExportCfg(BaseModel):
    input_shape: list[int] = Field(default_factory=lambda: [1, 3, 224, 224])
    opset: int = 17


class ModelCfg(BaseModel):
    """Everything the pipeline needs to know about one model's training run."""

    name: ModelName
    data: DataCfg = Field(default_factory=DataCfg)
    export: ExportCfg = Field(default_factory=ExportCfg)
    # Model-specific blocks kept raw; Anomalib consumes them directly.
    model: dict[str, Any] = Field(default_factory=dict)
    trainer: dict[str, Any] = Field(default_factory=dict)
    optim: dict[str, Any] = Field(default_factory=dict)


_CONFIG_FILES: dict[ModelName, str] = {
    "patchcore": "patchcore.yaml",
    "efficient_ad": "efficient_ad.yaml",
}


@lru_cache
def load_model_cfg(model: ModelName) -> ModelCfg:
    path = CONFIGS_DIR / _CONFIG_FILES[model]
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw.setdefault("name", model)
    return ModelCfg.model_validate(raw)


__all__ = ["DataCfg", "ExportCfg", "ModelCfg", "load_model_cfg"]
