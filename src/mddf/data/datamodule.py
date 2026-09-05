"""Build an Anomalib datamodule for one (model, category) pair.

Thin wrapper over ``anomalib.data.MVTecAD`` (Anomalib 2.6). Only the kwargs that
have been stable across Anomalib 2.x are passed. Image resizing / normalisation is
intentionally *not* set here — in Anomalib 2.x that belongs to the model's
``PreProcessor``, which the training code (M2) configures from
``configs/<model>.yaml`` so train and export use identical preprocessing.

Needs the ``train`` optional dependencies.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from mddf.config import ModelName
from mddf.data.download import dataset_dir
from mddf.logging import get_logger
from mddf.training.configs import load_model_cfg

if TYPE_CHECKING:
    from anomalib.data import AnomalibDataModule

_log = get_logger("mddf.data.datamodule")


def build_datamodule(
    category: str,
    model: ModelName,
    *,
    root: Path | None = None,
) -> AnomalibDataModule:
    """Instantiate ``MVTecAD`` for ``category`` with batch sizes from the model config.

    The datamodule is returned un-prepared; the caller runs ``prepare_data()`` /
    ``setup()`` (or hands it to ``anomalib.engine.Engine``).
    """
    from anomalib.data import MVTecAD

    cfg = load_model_cfg(model).data
    _log.info(
        "build_datamodule",
        model=model,
        category=category,
        train_batch_size=cfg.train_batch_size,
        eval_batch_size=cfg.eval_batch_size,
    )
    dm: AnomalibDataModule = MVTecAD(
        root=dataset_dir(root),
        category=category,
        train_batch_size=cfg.train_batch_size,
        eval_batch_size=cfg.eval_batch_size,
        num_workers=cfg.num_workers,
    )
    return dm


__all__ = ["build_datamodule"]
