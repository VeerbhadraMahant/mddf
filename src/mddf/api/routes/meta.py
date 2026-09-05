"""Catalogue and benchmark endpoints."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter

from mddf.api.schemas import (
    BenchmarkResponse,
    BenchmarkRow,
    CategoriesResponse,
    CategoryInfo,
    CategoryMetrics,
)
from mddf.catalog import load_catalog
from mddf.config import ALL_MODELS, ModelName
from mddf.inference.registry import get_registry
from mddf.reporting import generated_at, results_by_category

router = APIRouter(tags=["catalogue"])

_ALL_MODELS = ALL_MODELS


@router.get("/categories", response_model=CategoriesResponse)
async def categories() -> CategoriesResponse:
    catalog = load_catalog()
    results = results_by_category()
    try:
        exported = set(get_registry().available())
    except Exception:  # metadata endpoint must not 500 on artifact issues
        exported = set()

    infos: list[CategoryInfo] = []
    for category in catalog.categories:
        per_model = results.get(category.name, {})
        metrics: dict[ModelName, CategoryMetrics] = {}
        for model in _ALL_MODELS:
            m = per_model.get(model)
            if m is None:
                continue
            metrics[model] = CategoryMetrics(
                image_auroc=m.get("image_auroc"),
                pixel_auroc=m.get("pixel_auroc"),
                aupro=m.get("aupro"),
                f1_max=m.get("f1_max"),
                published_image_auroc=category.published_image_auroc,
            )
        available = [m for m in _ALL_MODELS if (m, category.name) in exported] or list(metrics)
        infos.append(
            CategoryInfo(
                name=category.name,
                kind=category.kind,
                available_models=available,
                metrics=metrics,
            )
        )
    return CategoriesResponse(dataset=catalog.dataset, categories=infos)


@router.get("/benchmark", response_model=BenchmarkResponse)
async def benchmark() -> BenchmarkResponse:
    catalog = load_catalog()
    results = results_by_category()
    rows: list[BenchmarkRow] = []
    for category in catalog.categories:
        per_model = results.get(category.name, {})
        for model in _ALL_MODELS:
            m = per_model.get(model)
            if m is None:
                continue
            rows.append(
                BenchmarkRow(
                    category=category.name,
                    model=model,
                    image_auroc=m.get("image_auroc", 0.0),
                    pixel_auroc=m.get("pixel_auroc"),
                    aupro=m.get("aupro"),
                    f1_max=m.get("f1_max"),
                    published_image_auroc=category.published_image_auroc,
                    latency_ms_p50=m.get("latency_ms_p50"),
                    latency_ms_p95=m.get("latency_ms_p95"),
                )
            )
    return BenchmarkResponse(
        generated_at=generated_at() or datetime.now(UTC).isoformat(),
        rows=rows,
    )
