"""Aggregate per-category ``metrics.json`` into the benchmark table the API serves.

Reads ``artifacts/<model>/<category>/metrics.json`` (written by M2 training) plus,
when present, ``artifacts/benchmark/latency.json`` (M3 latency), and writes the
combined ``artifacts/benchmark/metrics.json`` in the shape :mod:`mddf.reporting`
expects. Also renders a Markdown comparison against the published PatchCore
baselines in ``configs/categories.yaml``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mddf.catalog import load_catalog
from mddf.config import ModelName, get_settings
from mddf.training import artifacts as art

_METRIC_KEYS = ("image_auroc", "pixel_auroc", "aupro", "f1_max")


def _artifacts_dir(root: Path | None) -> Path:
    return root or get_settings().artifacts_dir


def load_latency(root: Path | None = None) -> dict[str, dict[str, dict[str, float]]]:
    path = _artifacts_dir(root) / "benchmark" / "latency.json"
    if not path.is_file():
        return {}
    data: dict[str, Any] = art.read_json(path)
    return data.get("results", {})  # type: ignore[no-any-return]


def aggregate(
    models: list[ModelName],
    categories: list[str] | None = None,
    *,
    root: Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    catalog = load_catalog()
    names = categories or catalog.names
    latency = load_latency(root)

    results: dict[str, dict[str, dict[str, float]]] = {}
    for category in names:
        per_model: dict[str, dict[str, float]] = {}
        for model in models:
            mpath = art.metrics_path(model, category, root=root)
            if not mpath.is_file():
                continue
            payload = art.read_json(mpath)
            entry: dict[str, float] = {
                k: float(v) for k, v in payload.get("metrics", {}).items() if k in _METRIC_KEYS
            }
            for tname, tval in payload.get("thresholds", {}).items():
                entry[f"threshold_{tname}"] = float(tval)
            lat = latency.get(category, {}).get(model, {})
            for k in ("latency_ms_p50", "latency_ms_p95", "latency_ms_mean"):
                if k in lat:
                    entry[k] = float(lat[k])
            if entry:
                per_model[model] = entry
        if per_model:
            results[category] = per_model

    doc: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "results": results,
    }
    if write:
        art.write_json(_artifacts_dir(root) / "benchmark" / "metrics.json", doc)
    return doc


def comparison_markdown(root: Path | None = None) -> str:
    catalog = load_catalog()
    doc = aggregate(["patchcore", "efficient_ad"], root=root, write=False)
    results: dict[str, Any] = doc["results"]

    lines = [
        "| Category | Model | Image AUROC | Published (PatchCore) | Δ | Pixel AUROC | AUPRO | "
        "CPU p50 (ms) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for category in catalog.categories:
        published = category.published_image_auroc
        for model, entry in results.get(category.name, {}).items():
            img = entry.get("image_auroc")
            delta = f"{img - published:+.3f}" if img is not None else "—"
            lines.append(
                f"| {category.name} | {model} | "
                f"{_fmt(img)} | {published:.3f} | {delta} | "
                f"{_fmt(entry.get('pixel_auroc'))} | {_fmt(entry.get('aupro'))} | "
                f"{_fmt(entry.get('latency_ms_p50'), 1)} |"
            )
    return "\n".join(lines)


def _fmt(value: float | None, places: int = 3) -> str:
    return f"{value:.{places}f}" if value is not None else "—"


__all__ = ["aggregate", "comparison_markdown", "load_latency"]
