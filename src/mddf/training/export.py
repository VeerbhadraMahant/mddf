"""Export trained checkpoints to ONNX for Torch-free CPU serving.

Uses Anomalib's ``Engine.export`` (ExportType.ONNX), which bakes the model's
``PreProcessor`` (resize + normalise) and post-processing into the graph. The
resulting ``model.onnx`` + ``preprocess.json`` are the only things the inference
service needs — no PyTorch at runtime.

A round-trip sanity check runs the exported graph once under ONNXRuntime and
records its output tensor names/shapes into ``export.json``.

Needs the ``train`` optional dependencies.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from mddf.config import ModelName
from mddf.logging import get_logger
from mddf.training import artifacts as art
from mddf.training.configs import load_model_cfg
from mddf.training.preprocess_spec import PreprocessSpec
from mddf.training.train import build_model, preprocess_spec_for

_log = get_logger("mddf.training.export")


@dataclass
class ExportResult:
    model: ModelName
    category: str
    onnx: Path
    spec: PreprocessSpec
    input_name: str
    output_names: list[str]
    output_shapes: list[list[int]]
    int8_onnx: Path | None = None
    int8_bytes: int | None = None


def quantize_int8(fp32_onnx: Path) -> Path:
    """Dynamic INT8 quantization (weights only, no calibration data needed).

    Shrinks the graph ~4x and speeds up CPU matmul-heavy models; accuracy impact
    is measured by ``mddf benchmark`` / ``mddf report`` re-scoring from the ONNX.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.shape_inference import quant_pre_process

    out = fp32_onnx.with_suffix(".int8.onnx")
    with tempfile.TemporaryDirectory() as tmp:
        prepped = Path(tmp) / "prepped.onnx"
        try:
            quant_pre_process(str(fp32_onnx), str(prepped))
            source = prepped
        except Exception:  # pre-process is best-effort
            source = fp32_onnx
        quantize_dynamic(str(source), str(out), weight_type=QuantType.QInt8)
    return out


def _onnx_signature(onnx_file: Path, network_input: tuple[int, int]) -> dict[str, Any]:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    h, w = network_input
    sample = np.zeros((1, 3, h, w), dtype=np.float32)
    outputs = sess.run(None, {inp.name: sample})
    return {
        "input_name": inp.name,
        "input_shape": list(inp.shape),
        "output_names": [o.name for o in sess.get_outputs()],
        "output_shapes": [list(np.asarray(o).shape) for o in outputs],
    }


def export_category(
    model: ModelName,
    category: str,
    *,
    output_root: Path | None = None,
    force: bool = False,
    quantize: bool = False,
) -> ExportResult:
    cfg = load_model_cfg(model)
    ckpt = art.checkpoint_path(model, category, root=output_root)
    if not ckpt.is_file():
        raise FileNotFoundError(f"No checkpoint for {model}/{category} at {ckpt}; train first.")

    onnx_dest = art.onnx_path(model, category, root=output_root)
    spec = preprocess_spec_for(model, cfg)

    if onnx_dest.is_file() and not force:
        _log.info("skip_existing_export", model=model, category=category)
    else:
        from anomalib.deploy import ExportType  # heavy; only needed for a real export
        from anomalib.engine import Engine

        net = build_model(model, cfg)
        engine = Engine(logger=False, accelerator="cpu", devices=1)
        with tempfile.TemporaryDirectory() as tmp:
            engine.export(
                model=net,
                export_type=ExportType.ONNX,
                export_root=tmp,
                input_size=spec.network_input,
                ckpt_path=str(ckpt),
            )
            produced = next(Path(tmp).rglob("*.onnx"), None)
            if produced is None:
                raise RuntimeError(f"Anomalib produced no .onnx for {model}/{category}")
            onnx_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(produced, onnx_dest)

    art.write_json(art.preprocess_path(model, category, root=output_root), spec.model_dump())
    sig = _onnx_signature(onnx_dest, spec.network_input)

    int8_path: Path | None = None
    int8_bytes: int | None = None
    if quantize:
        candidate = onnx_dest.with_suffix(".int8.onnx")
        if candidate.is_file() and not force:
            int8_path = candidate
        else:
            try:
                int8_path = quantize_int8(onnx_dest)
            except Exception:
                _log.exception("quantize_failed", model=model, category=category)
                int8_path = None
        if int8_path and int8_path.is_file():
            int8_bytes = int8_path.stat().st_size

    art.write_json(
        art.category_dir(model, category, root=output_root) / "export.json",
        {
            "model": model,
            "category": category,
            "exported_at": datetime.now(UTC).isoformat(),
            "onnx_bytes": onnx_dest.stat().st_size,
            "int8_bytes": int8_bytes,
            **sig,
        },
    )
    _log.info(
        "export_done",
        model=model,
        category=category,
        mb=round(onnx_dest.stat().st_size / 1e6, 1),
        int8_mb=round(int8_bytes / 1e6, 1) if int8_bytes else None,
        outputs=sig["output_names"],
    )
    return ExportResult(
        model=model,
        category=category,
        onnx=onnx_dest,
        spec=spec,
        input_name=sig["input_name"],
        output_names=sig["output_names"],
        output_shapes=sig["output_shapes"],
        int8_onnx=int8_path,
        int8_bytes=int8_bytes,
    )


def export_matrix(
    models: list[ModelName],
    categories: list[str],
    *,
    output_root: Path | None = None,
    force: bool = False,
    quantize: bool = False,
) -> tuple[list[ExportResult], list[tuple[str, str, str]]]:
    results: list[ExportResult] = []
    failures: list[tuple[str, str, str]] = []
    for model in models:
        for category in categories:
            try:
                results.append(
                    export_category(
                        model,
                        category,
                        output_root=output_root,
                        force=force,
                        quantize=quantize,
                    )
                )
            except Exception as exc:
                _log.exception("export_failed", model=model, category=category)
                failures.append((model, category, f"{type(exc).__name__}: {exc}"))
    return results, failures


__all__ = ["ExportResult", "export_category", "export_matrix"]
