"""Interpret Anomalib ONNX outputs into an anomaly map + image score.

Anomalib image models export some subset of ``anomaly_map``, ``pred_score``,
``pred_label``, ``pred_mask``. Names and ordering vary by model/version, so we
match by name (with fallbacks) and by array rank.
"""

from __future__ import annotations

import numpy as np

_MAP_KEYS = ("anomaly_map", "anomaly_maps", "heat_map", "map")
_SCORE_KEYS = ("pred_score", "pred_scores", "score", "image_score")


class RawPrediction:
    __slots__ = ("anomaly_map", "score")

    def __init__(self, anomaly_map: np.ndarray, score: float) -> None:
        self.anomaly_map = anomaly_map  # (H, W) float32
        self.score = score


def _first(outputs: dict[str, np.ndarray], keys: tuple[str, ...]) -> np.ndarray | None:
    lower = {k.lower(): v for k, v in outputs.items()}
    for key in keys:
        if key in lower:
            return lower[key]
    return None


def _to_hw(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    while a.ndim > 2:
        a = a[0]
    return a


def parse_outputs(outputs: dict[str, np.ndarray]) -> RawPrediction:
    amap = _first(outputs, _MAP_KEYS)
    score_arr = _first(outputs, _SCORE_KEYS)

    # Fall back positionally: the rank-4/3 tensor is the map, the scalar is the score.
    if amap is None:
        for v in outputs.values():
            if np.asarray(v).ndim >= 2:
                amap = v
                break
    if score_arr is None:
        for v in outputs.values():
            a = np.asarray(v)
            if a.ndim <= 1 and a.size == 1:
                score_arr = a
                break

    if amap is None:
        raise ValueError(f"No anomaly map in ONNX outputs: {list(outputs)}")

    amap_hw = _to_hw(amap)
    if score_arr is not None:
        score = float(np.asarray(score_arr).reshape(-1)[0])
    else:
        score = float(amap_hw.max())
    return RawPrediction(anomaly_map=amap_hw, score=score)


__all__ = ["RawPrediction", "parse_outputs"]
