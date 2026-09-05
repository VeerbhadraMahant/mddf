"""Render anomaly maps into operator-facing overlays, masks and boxes."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class BBoxData:
    x: int
    y: int
    width: int
    height: int
    score: float


def _to_original(map_hw: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    w, h = size_wh
    resized = cv2.resize(map_hw.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    return np.asarray(resized, dtype=np.float32)


def normalize_map(map_hw: np.ndarray, *, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    """Scale to [0, 1]. If the map already looks like [0, 1], keep it; otherwise
    stretch robust percentiles so the overlay has usable contrast."""
    m = np.asarray(map_hw, dtype=np.float32)
    if float(m.min()) >= 0.0 and float(m.max()) <= 1.0 + 1e-6:
        return np.clip(m, 0.0, 1.0).astype(np.float32)
    lo = float(np.percentile(m, lo_pct))
    hi = float(np.percentile(m, hi_pct))
    if hi - lo < 1e-6:
        return np.zeros_like(m)
    return np.clip((m - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def overlay(original_rgb: np.ndarray, map_hw: np.ndarray, *, alpha: float = 0.5) -> np.ndarray:
    h, w = original_rgb.shape[:2]
    norm = _to_original(normalize_map(map_hw), (w, h))
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat_rgb = np.asarray(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB), dtype=np.float32)
    base = original_rgb.astype(np.float32)
    blended = (1 - alpha) * base + alpha * heat_rgb
    return np.clip(blended, 0, 255).astype(np.uint8)


def binary_mask(
    map_hw: np.ndarray,
    size_wh: tuple[int, int],
    *,
    pixel_threshold: float | None = None,
) -> np.ndarray:
    norm = normalize_map(map_hw)
    full = _to_original(norm, size_wh)
    if pixel_threshold is not None and 0.0 <= pixel_threshold <= 1.0:
        thr = pixel_threshold
    else:
        thr = max(float(np.percentile(full, 99.0)), 0.5)
    return (full >= thr).astype(np.uint8) * 255


def bounding_boxes(
    mask: np.ndarray,
    map_hw: np.ndarray,
    *,
    min_area_frac: float = 0.0005,
    max_boxes: int = 10,
) -> list[BBoxData]:
    h, w = mask.shape[:2]
    min_area = max(int(min_area_frac * h * w), 4)
    score_full = _to_original(normalize_map(map_hw), (w, h))

    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    boxes: list[BBoxData] = []
    for i in range(1, count):
        x, y, bw, bh, area = stats[i]
        if area < min_area:
            continue
        region = score_full[y : y + bh, x : x + bw]
        boxes.append(
            BBoxData(
                x=int(x),
                y=int(y),
                width=int(bw),
                height=int(bh),
                score=float(region.max()) if region.size else 0.0,
            )
        )
    boxes.sort(key=lambda b: b.score, reverse=True)
    return boxes[:max_boxes]


def to_png(image: np.ndarray) -> bytes:
    if image.ndim == 2:
        ok, buf = cv2.imencode(".png", image)
    else:
        ok, buf = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return bytes(buf)


__all__ = [
    "BBoxData",
    "binary_mask",
    "bounding_boxes",
    "normalize_map",
    "overlay",
    "to_png",
]
