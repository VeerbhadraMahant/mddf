"""Unit tests for the training helpers that don't need a GPU or the dataset."""

from __future__ import annotations

from pathlib import Path

import pytest

from mddf.training import artifacts as art
from mddf.training.configs import load_model_cfg
from mddf.training.preprocess_spec import efficient_ad_spec, patchcore_spec
from mddf.training.run import resolve_categories
from mddf.training.train import (
    _normalise_metrics,
    preprocess_spec_for,
    train_category,
)


def test_metric_normalisation_maps_aliases() -> None:
    raw = {
        "image_AUROC": 0.991,
        "pixel_AUROC": 0.977,
        "image_F1Score": 0.95,
        "pixel_AUPRO": 0.93,
        "some_other_metric": 1.0,
    }
    out = _normalise_metrics(raw)
    assert out == {
        "image_auroc": 0.991,
        "pixel_auroc": 0.977,
        "f1_max": 0.95,
        "aupro": 0.93,
    }


def test_preprocess_specs() -> None:
    pc = patchcore_spec(256, 224)
    assert pc.network_input == (224, 224)
    assert pc.normalize is True
    ea = efficient_ad_spec(256)
    assert ea.network_input == (256, 256)
    assert ea.normalize is False


def test_preprocess_spec_for_reads_config() -> None:
    assert preprocess_spec_for("patchcore", load_model_cfg("patchcore")).center_crop == (224, 224)
    assert preprocess_spec_for("efficient_ad", load_model_cfg("efficient_ad")).center_crop is None


def test_resolve_categories_validates() -> None:
    assert resolve_categories(["leather", "grid"]) == ["leather", "grid"]
    assert len(resolve_categories(None)) == 15
    with pytest.raises(ValueError, match="Unknown categories"):
        resolve_categories(["leather", "bogus"])


def test_artifact_paths(tmp_path: Path) -> None:
    assert art.checkpoint_path("patchcore", "leather", root=tmp_path).name == "model.ckpt"
    assert art.onnx_path("patchcore", "leather", root=tmp_path).parent.name == "leather"
    art.write_json(art.metrics_path("patchcore", "leather", root=tmp_path), {"a": 1})
    assert art.read_json(art.metrics_path("patchcore", "leather", root=tmp_path)) == {"a": 1}


def test_train_category_resumes_from_existing_artifacts(tmp_path: Path) -> None:
    """If metrics.json + checkpoint already exist, training must be skipped entirely
    (no Anomalib import / GPU needed)."""
    art.checkpoint_path("patchcore", "leather", root=tmp_path).parent.mkdir(
        parents=True, exist_ok=True
    )
    art.checkpoint_path("patchcore", "leather", root=tmp_path).write_bytes(b"ckpt")
    art.write_json(
        art.metrics_path("patchcore", "leather", root=tmp_path),
        {
            "metrics": {"image_auroc": 0.994},
            "thresholds": {"image": 12.3},
            "seconds": 42.0,
            "raw_metrics": {},
        },
    )
    result = train_category("patchcore", "leather", output_root=tmp_path)
    assert result.metrics == {"image_auroc": 0.994}
    assert result.thresholds == {"image": 12.3}
    assert result.seconds == 42.0
