"""Read benchmark results produced by the training pipeline.

The training/benchmark stage writes ``artifacts/benchmark/metrics.json`` with the
shape::

    {
      "generated_at": "2026-09-05T12:00:00Z",
      "results": {
        "<category>": {
          "<model>": {
            "image_auroc": 0.994,
            "pixel_auroc": 0.981,
            "aupro": 0.945,
            "f1_max": 0.972,
            "threshold": 12.34,
            "latency_ms_p50": 41.2,
            "latency_ms_p95": 58.9
          }
        }
      }
    }

The service reads it if present and otherwise reports "no results yet" rather than
failing — useful during the milestones before any model is trained.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from mddf.config import get_settings

METRICS_RELATIVE = ("benchmark", "metrics.json")


@lru_cache
def _load_raw() -> dict[str, Any]:
    path = get_settings().artifacts_dir.joinpath(*METRICS_RELATIVE)
    if not path.is_file():
        return {}
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def clear_cache() -> None:
    _load_raw.cache_clear()


def results_by_category() -> dict[str, dict[str, dict[str, float]]]:
    """``{category: {model: {metric: value}}}`` — empty when nothing is trained."""
    raw = _load_raw()
    return raw.get("results", {})  # type: ignore[no-any-return]


def generated_at() -> str | None:
    return _load_raw().get("generated_at")


__all__ = ["clear_cache", "generated_at", "results_by_category"]
