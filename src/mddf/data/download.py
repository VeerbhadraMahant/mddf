"""Acquire and verify the MVTec AD dataset.

Downloading is delegated to Anomalib's ``MVTecAD`` datamodule (``prepare_data``),
which fetches and extracts the official archive with a hash check. On top of that we
run :func:`mddf.data.layout.inspect_dataset` to confirm every requested category is
present with the expected image counts before any training starts.

Needs the ``train`` optional dependencies (``pip install -e ".[train]"``).
"""

from __future__ import annotations

from pathlib import Path

from mddf.catalog import load_catalog
from mddf.config import get_settings
from mddf.data.layout import DatasetStatus, inspect_dataset
from mddf.logging import get_logger

_log = get_logger("mddf.data.download")

# Anomalib extracts here by default when given ``root=<dataset_root>``.
_SUBDIR = "MVTecAD"


def dataset_dir(root: Path | None = None) -> Path:
    base = root or get_settings().dataset_root
    return base / _SUBDIR


def ensure_categories(
    categories: list[str] | None = None,
    *,
    root: Path | None = None,
) -> DatasetStatus:
    """Download any missing categories, then verify the on-disk layout.

    Raises
    ------
    RuntimeError
        If, after the download attempt, any requested category is still missing or
        has unexpected image counts.
    """
    from anomalib.data import MVTecAD  # heavy; import lazily

    names = categories or load_catalog().names
    target = dataset_dir(root)
    target.mkdir(parents=True, exist_ok=True)

    for name in names:
        _log.info("prepare_category", category=name, root=str(target.parent))
        datamodule = MVTecAD(root=target, category=name)
        datamodule.prepare_data()

    status = inspect_dataset(target, names)
    for cat in status.categories:
        level = "info" if cat.ok else "error"
        getattr(_log, level)(
            "verify_category",
            category=cat.name,
            present=cat.present,
            train_good=cat.train_good,
            test_total=cat.test_total,
            defect_types=len(cat.defect_types),
            issues=cat.issues,
        )

    if not status.ok:
        broken = [c.name for c in status.categories if not c.ok]
        raise RuntimeError(
            f"MVTec AD verification failed for: {', '.join(broken)}. "
            f"Inspect {target} or delete it and re-run."
        )
    _log.info("dataset_ready", root=str(target), categories=len(status.categories))
    return status


__all__ = ["dataset_dir", "ensure_categories"]
