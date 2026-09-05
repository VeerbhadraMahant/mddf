"""Defect-detection endpoints: single image and batch."""

from __future__ import annotations

import base64

from fastapi import APIRouter, File, Form, UploadFile
from starlette.concurrency import run_in_threadpool

from mddf.api.errors import (
    artifacts_unavailable,
    invalid_image,
    payload_too_large,
    unknown_category,
    unsupported_media_type,
)
from mddf.api.schemas import (
    BatchPredictItem,
    BatchPredictResponse,
    BBox,
    PredictResponse,
)
from mddf.catalog import load_catalog
from mddf.config import ModelName, get_settings
from mddf.inference.predict import Prediction, run_prediction
from mddf.inference.registry import ArtifactsUnavailable

router = APIRouter(tags=["inference"])


def _check_media_type(file: UploadFile) -> None:
    allowed = get_settings().allowed_content_types
    if file.content_type not in allowed:
        raise unsupported_media_type(file.content_type, allowed)


async def _read_capped(file: UploadFile) -> bytes:
    limit = get_settings().max_upload_bytes
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise payload_too_large(limit)
    if not data:
        raise invalid_image("empty upload")
    return data


def _check_category(category: str) -> None:
    if category not in load_catalog().names:
        raise unknown_category(category, load_catalog().names)


def _to_response(pred: Prediction) -> PredictResponse:
    return PredictResponse(
        category=pred.category,
        model=pred.model,
        model_version=pred.model_version,
        verdict=pred.verdict,
        anomaly_score=pred.anomaly_score,
        normalized_score=pred.normalized_score,
        threshold=pred.threshold,
        latency_ms=pred.latency_ms,
        heatmap_png=base64.b64encode(pred.heatmap_png).decode("ascii"),
        mask_png=base64.b64encode(pred.mask_png).decode("ascii"),
        bboxes=[
            BBox(x=b.x, y=b.y, width=b.width, height=b.height, score=b.score) for b in pred.bboxes
        ],
    )


async def _predict_one(data: bytes, category: str, model: ModelName) -> PredictResponse:
    try:
        pred = await run_in_threadpool(run_prediction, data, category, model)
    except ArtifactsUnavailable as exc:
        raise artifacts_unavailable(f"No trained model for {model}/{category}: {exc}") from exc
    except (OSError, ValueError) as exc:
        raise invalid_image(str(exc)) from exc
    return _to_response(pred)


@router.post("/predict", response_model=PredictResponse)
async def predict(
    category: str = Form(...),
    model: ModelName = Form(default="patchcore"),
    file: UploadFile = File(...),
) -> PredictResponse:
    _check_media_type(file)
    _check_category(category)
    data = await _read_capped(file)
    return await _predict_one(data, category, model)


@router.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(
    category: str = Form(...),
    model: ModelName = Form(default="patchcore"),
    files: list[UploadFile] = File(...),
) -> BatchPredictResponse:
    _check_category(category)
    items: list[BatchPredictItem] = []
    for file in files:
        name = file.filename or "upload"
        try:
            _check_media_type(file)
            data = await _read_capped(file)
            result = await _predict_one(data, category, model)
            items.append(BatchPredictItem(filename=name, result=result))
        except Exception as exc:  # each item's error is reported inline
            detail = getattr(exc, "detail", None) or str(exc)
            items.append(BatchPredictItem(filename=name, error=detail))
    return BatchPredictResponse(category=category, model=model, items=items)


__all__ = ["router"]
