"""Train one (model, category) pair with Anomalib's ``Engine``.

PatchCore fits its coreset memory bank in a single pass; EfficientAD runs a
step-budgeted student-teacher schedule. Both are trained on defect-free images
only. Metrics + provenance are written under ``artifacts/<model>/<category>/``.

Needs the ``train`` optional dependencies and (in practice) an NVIDIA GPU.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mddf.config import ModelName
from mddf.data.datamodule import build_datamodule
from mddf.data.download import dataset_dir
from mddf.logging import get_logger
from mddf.training import artifacts as art
from mddf.training.configs import ModelCfg, load_model_cfg
from mddf.training.preprocess_spec import (
    PreprocessSpec,
    efficient_ad_spec,
    padim_spec,
    patchcore_spec,
)

_log = get_logger("mddf.training.train")

# Anomalib test-metric keys vary with the evaluator config; map what we recognise
# onto stable names.
_METRIC_ALIASES: dict[str, str] = {
    "image_auroc": "image_auroc",
    "auroc": "image_auroc",
    "image_auc": "image_auroc",
    "pixel_auroc": "pixel_auroc",
    "pixel_auc": "pixel_auroc",
    "image_f1score": "f1_max",
    "f1score": "f1_max",
    "image_f1": "f1_max",
    "pixel_aupro": "aupro",
    "aupro": "aupro",
    "pixel_pro": "aupro",
}


@dataclass
class TrainResult:
    model: ModelName
    category: str
    metrics: dict[str, float]
    thresholds: dict[str, float]
    checkpoint: Path
    run_dir: Path
    seconds: float
    raw_metrics: dict[str, float] = field(default_factory=dict)


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def preprocess_spec_for(model: ModelName, cfg: ModelCfg) -> PreprocessSpec:
    if model == "patchcore":
        crop = cfg.data.center_crop or 224
        return patchcore_spec(cfg.data.image_size, crop)
    if model == "padim":
        return padim_spec(cfg.data.image_size)
    return efficient_ad_spec(cfg.data.image_size)


def _build_evaluator() -> Any:
    """Test metrics: image + pixel AUROC, pixel AUPRO (the MVTec localization
    metric), image F1. ``strict=False`` so a category without masks still runs."""
    from anomalib.metrics import AUPRO, AUROC, Evaluator, F1Score

    test_metrics = [
        AUROC(fields=["pred_score", "gt_label"], prefix="image_", strict=False),
        AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_", strict=False),
        AUPRO(fields=["anomaly_map", "gt_mask"], prefix="pixel_", strict=False),
        F1Score(fields=["pred_label", "gt_label"], prefix="image_", strict=False),
    ]
    return Evaluator(test_metrics=test_metrics, compute_on_cpu=True)


def build_model(model: ModelName, cfg: ModelCfg) -> Any:
    from anomalib.models import EfficientAd, Padim, Patchcore

    spec = preprocess_spec_for(model, cfg)
    m = cfg.model
    evaluator = _build_evaluator()

    if model == "patchcore":
        pre = Patchcore.configure_pre_processor(
            image_size=spec.image_size, center_crop_size=spec.center_crop
        )
        return Patchcore(
            backbone=m.get("backbone", "wide_resnet50_2"),
            layers=tuple(m.get("layers", ["layer2", "layer3"])),
            pre_trained=m.get("pre_trained", True),
            coreset_sampling_ratio=m.get("coreset_sampling_ratio", 0.1),
            num_neighbors=m.get("num_neighbors", 9),
            pre_processor=pre,
            evaluator=evaluator,
        )

    if model == "padim":
        pre = Padim.configure_pre_processor(image_size=spec.image_size)
        return Padim(
            backbone=m.get("backbone", "resnet18"),
            layers=list(m.get("layers", ["layer1", "layer2", "layer3"])),
            pre_trained=m.get("pre_trained", True),
            n_features=m.get("n_features"),
            pre_processor=pre,
            evaluator=evaluator,
        )

    pre = EfficientAd.configure_pre_processor(image_size=spec.image_size)
    o = cfg.optim
    return EfficientAd(
        teacher_out_channels=m.get("teacher_out_channels", 384),
        model_size=m.get("model_size", "small"),
        pad_maps=m.get("pad_maps", True),
        lr=o.get("lr", 1e-4),
        weight_decay=o.get("weight_decay", 1e-5),
        pre_processor=pre,
        evaluator=evaluator,
    )


def build_engine(
    model: ModelName,
    cfg: ModelCfg,
    run_dir: Path,
    *,
    accelerator: str,
    max_steps: int | None = None,
) -> Any:
    from anomalib.engine import Engine

    # These are forwarded to the Lightning Trainer; `default_root_dir` / `logger`
    # are explicit Engine params and must not go through kwargs.
    trainer_kw: dict[str, Any] = {
        "accelerator": accelerator,
        "devices": 1,
        "enable_checkpointing": True,
    }
    if model in ("patchcore", "padim"):
        trainer_kw["max_epochs"] = 1
    else:
        trainer_kw["max_steps"] = int(
            max_steps if max_steps is not None else cfg.trainer.get("max_steps", 24000)
        )
        # EfficientAD uses batch size 1, so an epoch is only a few hundred batches;
        # an int val_check_interval would have to be smaller than that. Only honour
        # a float in (0, 1] (fraction of an epoch); otherwise validate per epoch.
        vci = cfg.trainer.get("val_check_interval")
        if isinstance(vci, float) and 0.0 < vci <= 1.0:
            trainer_kw["val_check_interval"] = vci
    if "precision" in cfg.trainer:
        trainer_kw["precision"] = cfg.trainer["precision"]
    return Engine(default_root_dir=str(run_dir), logger=False, **trainer_kw)


def _normalise_metrics(raw: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in raw.items():
        alias = _METRIC_ALIASES.get(key.lower().replace("test_", "").replace("/", "_"))
        if alias and alias not in out:
            out[alias] = float(value)
    return out


def _extract_thresholds(model_obj: Any) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    post = getattr(model_obj, "post_processor", None)
    for attr, name in (("image_threshold", "image"), ("pixel_threshold", "pixel")):
        val = getattr(post, attr, None)
        try:
            if val is not None:
                thresholds[name] = float(getattr(val, "value", val))
        except (TypeError, ValueError):
            continue
    return thresholds


def train_category(
    model: ModelName,
    category: str,
    *,
    dataset_root: Path | None = None,
    output_root: Path | None = None,
    accelerator: str = "auto",
    force: bool = False,
    max_steps: int | None = None,
) -> TrainResult:
    cfg = load_model_cfg(model)
    ckpt_dest = art.checkpoint_path(model, category, root=output_root)
    metrics_dest = art.metrics_path(model, category, root=output_root)

    if metrics_dest.is_file() and ckpt_dest.is_file() and not force:
        _log.info("skip_existing", model=model, category=category)
        payload = art.read_json(metrics_dest)
        return TrainResult(
            model=model,
            category=category,
            metrics=payload.get("metrics", {}),
            thresholds=payload.get("thresholds", {}),
            checkpoint=ckpt_dest,
            run_dir=ckpt_dest.parent.parent,
            seconds=payload.get("seconds", 0.0),
            raw_metrics=payload.get("raw_metrics", {}),
        )

    run_dir = art.category_dir(model, category, root=output_root) / "_run"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    datamodule = build_datamodule(category, model, root=dataset_root)
    net = build_model(model, cfg)
    engine = build_engine(model, cfg, run_dir, accelerator=accelerator, max_steps=max_steps)

    _log.info("fit_start", model=model, category=category)
    start = time.perf_counter()
    engine.fit(model=net, datamodule=datamodule)
    test_out = engine.test(model=net, datamodule=datamodule, verbose=False)
    seconds = time.perf_counter() - start

    raw: dict[str, float] = {}
    for row in test_out or []:
        raw.update({k: float(v) for k, v in row.items()})
    metrics = _normalise_metrics(raw)
    thresholds = _extract_thresholds(net)

    best = engine.trainer.checkpoint_callback.best_model_path if engine.trainer else ""
    src = Path(best) if best else next(run_dir.rglob("*.ckpt"), None)
    if src is None or not Path(src).is_file():
        raise RuntimeError(f"No checkpoint produced for {model}/{category} under {run_dir}")
    ckpt_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, ckpt_dest)

    art.write_json(
        metrics_dest,
        {
            "model": model,
            "category": category,
            "metrics": metrics,
            "thresholds": thresholds,
            "raw_metrics": raw,
            "seconds": round(seconds, 1),
        },
    )
    art.write_json(
        art.run_path(model, category, root=output_root),
        {
            "model": model,
            "category": category,
            "finished_at": datetime.now(UTC).isoformat(),
            "seconds": round(seconds, 1),
            "config": cfg.model_dump(),
            "git_sha": _git_sha(),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    )
    art.write_json(
        art.preprocess_path(model, category, root=output_root),
        preprocess_spec_for(model, cfg).model_dump(),
    )
    shutil.rmtree(run_dir, ignore_errors=True)

    _log.info("fit_done", model=model, category=category, seconds=round(seconds, 1), **metrics)
    return TrainResult(
        model=model,
        category=category,
        metrics=metrics,
        thresholds=thresholds,
        checkpoint=ckpt_dest,
        run_dir=ckpt_dest.parent.parent,
        seconds=seconds,
        raw_metrics=raw,
    )


def resolve_dataset_root(dataset_root: Path | None) -> Path:
    return dataset_dir(dataset_root)


__all__ = ["TrainResult", "build_engine", "build_model", "train_category"]
