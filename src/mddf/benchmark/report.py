"""Run every exported ONNX over its category's MVTec AD test split and derive
operating points (see :mod:`mddf.benchmark.operating_point`).

Self-contained: uses the exported ``model.onnx`` + local ``datasets/MVTecAD`` — no
PyTorch, no retraining. Output: ``artifacts/report/operating_points.json`` and a
Markdown summary.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from mddf.benchmark.operating_point import operating_points, point_to_dict
from mddf.catalog import load_catalog
from mddf.config import ALL_MODELS, ModelName, get_settings
from mddf.data.download import dataset_dir
from mddf.data.layout import find_category_dir
from mddf.inference.onnx_session import OnnxModel
from mddf.inference.postprocess import parse_outputs
from mddf.inference.preprocess import preprocess
from mddf.logging import get_logger
from mddf.training import artifacts as art
from mddf.training.preprocess_spec import PreprocessSpec

_log = get_logger("mddf.benchmark.report")
_IMG_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def _score_test_split(
    onnx_file: Path, spec: PreprocessSpec, category_dir: Path
) -> tuple[np.ndarray, np.ndarray]:
    session = OnnxModel(onnx_file, threads=1)
    scores: list[float] = []
    labels: list[int] = []
    for sub in sorted((category_dir / "test").iterdir()):
        if not sub.is_dir():
            continue
        is_defect = 0 if sub.name == "good" else 1
        for img in sorted(sub.iterdir()):
            if img.suffix.lower() not in _IMG_SUFFIXES:
                continue
            decoded = preprocess(img.read_bytes(), spec)
            raw = parse_outputs(session.run(decoded.tensor))
            scores.append(raw.score)
            labels.append(is_defect)
    return np.asarray(scores), np.asarray(labels)


def build_report(
    models: list[ModelName] | None = None,
    categories: list[str] | None = None,
    *,
    root: Path | None = None,
    dataset_root: Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    mods = models or list(ALL_MODELS)
    cats = categories or load_catalog().names
    ds = dataset_dir(dataset_root)

    results: dict[str, dict[str, Any]] = {}
    for category in cats:
        cdir = find_category_dir(ds, category)
        if cdir is None:
            continue
        for model in mods:
            onnx_file = art.onnx_path(model, category, root=root)
            spec_file = art.preprocess_path(model, category, root=root)
            if not onnx_file.is_file() or not spec_file.is_file():
                continue
            spec = PreprocessSpec.model_validate(art.read_json(spec_file))
            scores, labels = _score_test_split(onnx_file, spec, cdir)
            pts = operating_points(scores, labels)
            results.setdefault(category, {})[model] = {
                "n_good": int((labels == 0).sum()),
                "n_defect": int((labels == 1).sum()),
                "operating_points": [point_to_dict(p) for p in pts],
            }
            _log.info(
                "operating_points",
                model=model,
                category=category,
                points={p.name: round(p.false_alarm_rate, 3) for p in pts},
            )

    doc: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "results": results,
    }
    if write:
        out = (root or get_settings().artifacts_dir) / "report"
        art.write_json(out / "operating_points.json", doc)
        (out / "OPERATING_POINTS.md").write_text(_markdown(doc), encoding="utf-8")
    return doc


def _markdown(doc: dict[str, Any]) -> str:
    lines = [
        "# Operating points",
        "",
        "False-alarm rate (FPR) = defect-free parts wrongly rejected. "
        "Miss rate (FNR) = defects passed through.",
        "",
        "| Category | Model | Target | Threshold | Recall | Precision | False alarm | Miss |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for category, per_model in doc["results"].items():
        for model, payload in per_model.items():
            for p in payload["operating_points"]:
                lines.append(
                    f"| {category} | {model} | {p['name']} | {p['threshold']:.3f} | "
                    f"{p['recall']:.3f} | {p['precision']:.3f} | "
                    f"{p['false_alarm_rate']:.3f} | {p['miss_rate']:.3f} |"
                )
    return "\n".join(lines) + "\n"


__all__ = ["build_report"]
