"""Resumable orchestration of training across models x categories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mddf.catalog import load_catalog
from mddf.config import ALL_MODELS, ModelName
from mddf.logging import get_logger
from mddf.training.train import TrainResult, train_category

_log = get_logger("mddf.training.run")


@dataclass
class MatrixOutcome:
    results: list[TrainResult]
    failures: list[tuple[ModelName, str, str]]  # (model, category, error)

    @property
    def ok(self) -> bool:
        return not self.failures


def resolve_categories(names: list[str] | None) -> list[str]:
    known = load_catalog().names
    if not names:
        return known
    unknown = [n for n in names if n not in known]
    if unknown:
        raise ValueError(f"Unknown categories: {', '.join(unknown)}")
    return names


def train_matrix(
    models: list[ModelName],
    categories: list[str] | None = None,
    *,
    dataset_root: Path | None = None,
    output_root: Path | None = None,
    accelerator: str = "auto",
    force: bool = False,
    max_steps: int | None = None,
) -> MatrixOutcome:
    cats = resolve_categories(categories)
    results: list[TrainResult] = []
    failures: list[tuple[ModelName, str, str]] = []

    for model in models:
        for category in cats:
            try:
                results.append(
                    train_category(
                        model,
                        category,
                        dataset_root=dataset_root,
                        output_root=output_root,
                        accelerator=accelerator,
                        force=force,
                        max_steps=max_steps,
                    )
                )
            except Exception as exc:  # keep going; one bad category shouldn't sink the run
                _log.exception("train_failed", model=model, category=category)
                failures.append((model, category, f"{type(exc).__name__}: {exc}"))

    return MatrixOutcome(results=results, failures=failures)


__all__ = ["ALL_MODELS", "MatrixOutcome", "resolve_categories", "train_matrix"]
