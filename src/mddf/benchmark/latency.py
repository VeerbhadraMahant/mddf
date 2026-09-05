"""Measure CPU inference latency of the exported ONNX models.

Runs ONNXRuntime on random inputs (shape from ``preprocess.json``) and records
p50 / p95 / mean milliseconds per image. Written to
``artifacts/benchmark/latency.json`` and folded into the benchmark table by
:mod:`mddf.benchmark.accuracy`.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from mddf.catalog import load_catalog
from mddf.config import ModelName, get_settings
from mddf.logging import get_logger
from mddf.training import artifacts as art
from mddf.training.preprocess_spec import PreprocessSpec

_log = get_logger("mddf.benchmark.latency")


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q))


def measure_onnx(
    onnx_file: Path,
    input_shape: tuple[int, int, int, int],
    *,
    runs: int = 50,
    warmup: int = 5,
    threads: int = 1,
) -> dict[str, float]:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(
        str(onnx_file), sess_options=opts, providers=["CPUExecutionProvider"]
    )
    input_name = sess.get_inputs()[0].name
    rng = np.random.default_rng(0)
    sample = rng.standard_normal(input_shape).astype(np.float32)

    for _ in range(warmup):
        sess.run(None, {input_name: sample})

    timings: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        sess.run(None, {input_name: sample})
        timings.append((time.perf_counter() - start) * 1000.0)

    return {
        "latency_ms_p50": round(_percentile(timings, 50), 2),
        "latency_ms_p95": round(_percentile(timings, 95), 2),
        "latency_ms_mean": round(float(np.mean(timings)), 2),
        "runs": runs,
        "threads": threads,
    }


def benchmark_all(
    models: list[ModelName],
    categories: list[str] | None = None,
    *,
    root: Path | None = None,
    runs: int = 50,
    write: bool = True,
) -> dict[str, Any]:
    names = categories or load_catalog().names
    results: dict[str, dict[str, dict[str, float]]] = {}
    for category in names:
        for model in models:
            onnx_file = art.onnx_path(model, category, root=root)
            spec_file = art.preprocess_path(model, category, root=root)
            if not onnx_file.is_file() or not spec_file.is_file():
                continue
            spec = PreprocessSpec.model_validate(art.read_json(spec_file))
            h, w = spec.network_input
            stats = measure_onnx(onnx_file, (1, 3, h, w), runs=runs)

            int8_file = onnx_file.with_suffix(".int8.onnx")
            if int8_file.is_file():
                q = measure_onnx(int8_file, (1, 3, h, w), runs=runs)
                stats["latency_ms_p50_int8"] = q["latency_ms_p50"]
                stats["latency_ms_p95_int8"] = q["latency_ms_p95"]
                stats["speedup_int8"] = round(
                    stats["latency_ms_p50"] / max(q["latency_ms_p50"], 1e-6), 2
                )

            results.setdefault(category, {})[model] = stats
            _log.info("latency", model=model, category=category, **stats)

    doc: dict[str, Any] = {"results": results}
    if write:
        base = root or get_settings().artifacts_dir
        art.write_json(base / "benchmark" / "latency.json", doc)
    return doc


__all__ = ["benchmark_all", "measure_onnx"]
