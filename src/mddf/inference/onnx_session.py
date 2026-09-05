"""Minimal ONNXRuntime wrapper: single-threaded CPU inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class OnnxModel:
    def __init__(self, onnx_path: Path, *, threads: int = 1) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._sess = ort.InferenceSession(
            str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.input_name: str = self._sess.get_inputs()[0].name
        self.output_names: list[str] = [o.name for o in self._sess.get_outputs()]

    def run(self, tensor: np.ndarray) -> dict[str, np.ndarray]:
        outputs: list[Any] = self._sess.run(None, {self.input_name: tensor})
        return {
            name: np.asarray(value) for name, value in zip(self.output_names, outputs, strict=True)
        }


__all__ = ["OnnxModel"]
