"""End-to-end single-image prediction (Torch-free)."""

from __future__ import annotations

import time
from dataclasses import dataclass

from mddf.config import ModelName
from mddf.inference import heatmap as hm
from mddf.inference.postprocess import parse_outputs
from mddf.inference.preprocess import preprocess
from mddf.inference.registry import LoadedModel, Registry, get_registry


@dataclass
class BBox:
    x: int
    y: int
    width: int
    height: int
    score: float


@dataclass
class Prediction:
    category: str
    model: ModelName
    model_version: str
    verdict: str
    anomaly_score: float
    normalized_score: float
    threshold: float
    latency_ms: float
    heatmap_png: bytes
    mask_png: bytes
    bboxes: list[BBox]
    image_auroc: float | None = None


def _predict_with(loaded: LoadedModel, image_bytes: bytes) -> Prediction:
    start = time.perf_counter()

    decoded = preprocess(image_bytes, loaded.spec)
    raw = parse_outputs(loaded.session.run(decoded.tensor))

    score = raw.score
    verdict = loaded.threshold.verdict(score)
    normalized = loaded.threshold.normalize(score)

    size_wh = decoded.original_size
    pixel_thr = None  # Anomalib bakes pixel post-processing; map is already ~[0,1]
    overlay_img = hm.overlay(decoded.array, raw.anomaly_map, alpha=0.5)
    mask = hm.binary_mask(raw.anomaly_map, size_wh, pixel_threshold=pixel_thr)
    boxes = [
        BBox(x=b.x, y=b.y, width=b.width, height=b.height, score=b.score)
        for b in hm.bounding_boxes(mask, raw.anomaly_map)
    ]
    latency_ms = (time.perf_counter() - start) * 1000.0

    return Prediction(
        category=loaded.category,
        model=loaded.model,
        model_version=loaded.version,
        verdict=verdict,
        anomaly_score=float(score),
        normalized_score=float(normalized),
        threshold=float(loaded.threshold.threshold),
        latency_ms=round(latency_ms, 2),
        heatmap_png=hm.to_png(overlay_img),
        mask_png=hm.to_png(mask),
        bboxes=boxes,
        image_auroc=loaded.metrics.get("image_auroc"),
    )


def run_prediction(
    image_bytes: bytes,
    category: str,
    model: ModelName,
    *,
    registry: Registry | None = None,
) -> Prediction:
    reg = registry or get_registry()
    loaded = reg.get(model, category)
    return _predict_with(loaded, image_bytes)


__all__ = ["BBox", "Prediction", "run_prediction"]
