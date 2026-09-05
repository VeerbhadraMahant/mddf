"""Pydantic v2 request/response models for the public API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from mddf.config import ModelName

Verdict = Literal["normal", "defect"]


class _Model(BaseModel):
    # Several responses carry a ``model`` / ``model_version`` field; opt out of
    # pydantic's protected "model_" namespace so those names are allowed.
    model_config = ConfigDict(protected_namespaces=())


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str
    version: str


class ReadyResponse(BaseModel):
    ready: bool
    detail: str


class CategoryMetrics(BaseModel):
    image_auroc: float | None = None
    pixel_auroc: float | None = None
    aupro: float | None = None
    f1_max: float | None = None
    published_image_auroc: float


class CategoryInfo(BaseModel):
    name: str
    kind: str
    available_models: list[ModelName]
    metrics: dict[ModelName, CategoryMetrics]


class CategoriesResponse(BaseModel):
    dataset: str
    categories: list[CategoryInfo]


class BBox(BaseModel):
    x: int
    y: int
    width: int
    height: int
    score: float


class PredictResponse(_Model):
    category: str
    model: ModelName
    model_version: str
    verdict: Verdict
    anomaly_score: float = Field(description="Raw image-level anomaly score.")
    normalized_score: float = Field(
        ge=0.0, le=1.0, description="Score mapped to [0, 1] with 0.5 at the decision threshold."
    )
    threshold: float
    latency_ms: float
    heatmap_png: str = Field(description="Base64 PNG: input with anomaly heatmap overlay.")
    mask_png: str = Field(description="Base64 PNG: binary anomaly mask.")
    bboxes: list[BBox]


class BatchPredictItem(BaseModel):
    filename: str
    result: PredictResponse | None = None
    error: str | None = None


class BatchPredictResponse(_Model):
    category: str
    model: ModelName
    items: list[BatchPredictItem]


class BenchmarkRow(_Model):
    category: str
    model: ModelName
    image_auroc: float
    pixel_auroc: float | None = None
    aupro: float | None = None
    f1_max: float | None = None
    published_image_auroc: float
    latency_ms_p50: float | None = None
    latency_ms_p95: float | None = None


class BenchmarkResponse(BaseModel):
    generated_at: str
    rows: list[BenchmarkRow]
