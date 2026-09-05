"""Parity gate: the INT8 export must match fp32 within tolerance.

For every ``(model, category)`` that has both ``model.onnx`` and
``model.int8.onnx``, re-score the MVTec AD test split with each and compare
image-level AUROC. Fails (exit 1 via the CLI) if any delta exceeds the tolerance,
so "INT8 is safe" is a checked claim rather than an assumption.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mddf.benchmark.operating_point import auroc
from mddf.benchmark.report import _score_test_split
from mddf.catalog import load_catalog
from mddf.config import ALL_MODELS, ModelName, get_settings
from mddf.data.download import dataset_dir
from mddf.data.layout import find_category_dir
from mddf.logging import get_logger
from mddf.training import artifacts as art
from mddf.training.preprocess_spec import PreprocessSpec

_log = get_logger("mddf.benchmark.verify")

DEFAULT_TOLERANCE = 0.01


def verify_int8_parity(
    models: list[ModelName] | None = None,
    categories: list[str] | None = None,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    root: Path | None = None,
    dataset_root: Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    mods = models or list(ALL_MODELS)
    cats = categories or load_catalog().names
    ds = dataset_dir(dataset_root)

    rows: list[dict[str, Any]] = []
    for category in cats:
        cdir = find_category_dir(ds, category)
        if cdir is None:
            continue
        for model in mods:
            fp32 = art.onnx_path(model, category, root=root)
            int8 = fp32.with_suffix(".int8.onnx")
            spec_file = art.preprocess_path(model, category, root=root)
            if not (fp32.is_file() and int8.is_file() and spec_file.is_file()):
                continue
            spec = PreprocessSpec.model_validate(art.read_json(spec_file))
            s_fp32, labels = _score_test_split(fp32, spec, cdir)
            s_int8, _ = _score_test_split(int8, spec, cdir)
            a32 = auroc(s_fp32, labels)
            a8 = auroc(s_int8, labels)
            delta = a8 - a32
            ok = abs(delta) <= tolerance
            rows.append(
                {
                    "model": model,
                    "category": category,
                    "auroc_fp32": round(a32, 4),
                    "auroc_int8": round(a8, 4),
                    "delta": round(delta, 4),
                    "within_tolerance": ok,
                }
            )
            (_log.info if ok else _log.error)(
                "int8_parity",
                model=model,
                category=category,
                auroc_fp32=round(a32, 4),
                auroc_int8=round(a8, 4),
                delta=round(delta, 4),
            )

    doc: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "tolerance": tolerance,
        "passed": all(r["within_tolerance"] for r in rows) if rows else True,
        "rows": rows,
    }
    if write:
        art.write_json((root or get_settings().artifacts_dir) / "report" / "parity.json", doc)
    return doc


__all__ = ["DEFAULT_TOLERANCE", "verify_int8_parity"]
