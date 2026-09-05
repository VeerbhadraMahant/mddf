"""Torch-free inference-core tests using a fake ONNX session."""

from __future__ import annotations

import io
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from PIL import Image

from mddf.inference import heatmap as hm
from mddf.inference.postprocess import parse_outputs
from mddf.inference.predict import run_prediction
from mddf.inference.preprocess import preprocess
from mddf.inference.registry import LoadedModel, Registry
from mddf.inference.threshold import build as build_threshold
from mddf.training.preprocess_spec import efficient_ad_spec, patchcore_spec


def _png(w: int = 120, h: int = 90, colour: int = 128) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (colour, colour, colour)).save(buf, format="PNG")
    return buf.getvalue()


# --- preprocess -----------------------------------------------------------


def test_preprocess_patchcore_shapes_and_range() -> None:
    decoded = preprocess(_png(), patchcore_spec(256, 224))
    assert decoded.tensor.shape == (1, 3, 224, 224)
    assert decoded.tensor.dtype == np.float32
    assert float(decoded.tensor.min()) >= 0.0 and float(decoded.tensor.max()) <= 1.0
    assert decoded.original_size == (120, 90)


def test_preprocess_efficient_ad_no_crop() -> None:
    decoded = preprocess(_png(), efficient_ad_spec(256))
    assert decoded.tensor.shape == (1, 3, 256, 256)


def test_preprocess_manual_normalise_when_not_baked() -> None:
    spec = patchcore_spec(256, 224).model_copy(update={"baked_into_onnx": False})
    baked = preprocess(_png(colour=10), patchcore_spec(256, 224))
    normed = preprocess(_png(colour=10), spec)
    # baked spec leaves values in [0,1]; manual normalisation pushes a dark image negative
    assert float(baked.tensor.min()) >= 0.0
    assert float(normed.tensor.min()) < -1.0


# --- postprocess --------------------------------------------------------


def test_parse_outputs_named() -> None:
    out = {"anomaly_map": np.zeros((1, 1, 8, 8), np.float32), "pred_score": np.array([0.7])}
    raw = parse_outputs(out)
    assert raw.anomaly_map.shape == (8, 8)
    assert raw.score == pytest.approx(0.7)


def test_parse_outputs_positional_fallback() -> None:
    out = {"x": np.full((1, 4, 4), 0.3, np.float32), "y": np.array(0.9, np.float32)}
    raw = parse_outputs(out)
    assert raw.anomaly_map.shape == (4, 4)
    assert raw.score == pytest.approx(0.9)


def test_parse_outputs_requires_a_map() -> None:
    with pytest.raises(ValueError, match="No anomaly map"):
        parse_outputs({"pred_score": np.array([0.1])})


# --- threshold --------------------------------------------------------


def test_threshold_verdict_and_normalisation() -> None:
    tm = build_threshold(10.0, ref_scores=(0.0, 20.0))
    assert tm.verdict(12.0) == "defect"
    assert tm.verdict(8.0) == "normal"
    assert tm.normalize(10.0) == pytest.approx(0.5, abs=1e-6)
    assert tm.normalize(100.0) > 0.99
    assert tm.normalize(-100.0) < 0.01


def test_threshold_defaults_without_value() -> None:
    tm = build_threshold(None)
    assert tm.threshold == 0.5


# --- heatmap --------------------------------------------------------


def test_heatmap_overlay_mask_bboxes() -> None:
    rng = np.random.default_rng(0)
    original = (rng.random((60, 80, 3)) * 255).astype(np.uint8)
    amap = np.zeros((32, 32), np.float32)
    amap[8:16, 20:28] = 1.0  # one hot blob

    over = hm.overlay(original, amap, alpha=0.5)
    assert over.shape == original.shape and over.dtype == np.uint8

    mask = hm.binary_mask(amap, (80, 60))
    assert mask.shape == (60, 80)
    assert mask.max() == 255

    boxes = hm.bounding_boxes(mask, amap)
    assert len(boxes) == 1
    assert boxes[0].width > 0 and boxes[0].score == pytest.approx(1.0, abs=1e-3)
    assert isinstance(hm.to_png(over), bytes)
    assert isinstance(hm.to_png(mask), bytes)


def test_normalize_map_passes_through_unit_range() -> None:
    m = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
    np.testing.assert_allclose(hm.normalize_map(m), m, atol=1e-6)


# --- registry LRU -------------------------------------------------------


class _FakeSession:
    output_names: ClassVar[list[str]] = ["anomaly_map", "pred_score"]

    def run(self, tensor: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "anomaly_map": np.zeros((1, 1, 16, 16), np.float32),
            "pred_score": np.array([0.42], np.float32),
        }


def _fake_loaded(model: str, category: str) -> LoadedModel:
    return LoadedModel(
        model=model,  # type: ignore[arg-type]
        category=category,
        session=_FakeSession(),  # type: ignore[arg-type]
        spec=patchcore_spec(64, 64),
        threshold=build_threshold(0.5),
        metrics={"image_auroc": 0.99},
        version="test",
    )


def test_registry_prefers_int8_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mddf.inference.registry import _resolve_files
    from mddf.training import artifacts as art

    cdir = art.category_dir("patchcore", "leather", root=tmp_path)
    cdir.mkdir(parents=True)
    (cdir / "model.onnx").write_bytes(b"fp32")
    (cdir / "model.int8.onnx").write_bytes(b"int8")
    (cdir / "preprocess.json").write_text("{}")
    (cdir / "metrics.json").write_text("{}")

    reg = Registry()
    monkeypatch.setattr(reg._settings, "artifacts_dir", tmp_path)
    monkeypatch.setattr(reg._settings, "hf_model_repo", "")

    def picked() -> str:
        return _resolve_files("patchcore", "leather", reg._settings)["model.onnx"].name

    monkeypatch.setattr(reg._settings, "prefer_int8", True)
    assert picked() == "model.int8.onnx"
    monkeypatch.setattr(reg._settings, "prefer_int8", False)
    assert picked() == "model.onnx"


def test_registry_lru_eviction(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = Registry()
    monkeypatch.setattr(reg._settings, "registry_cache_size", 2)
    monkeypatch.setattr(reg, "_load", _fake_loaded)

    reg.get("patchcore", "a")
    reg.get("patchcore", "b")
    reg.get("patchcore", "c")  # evicts "a"
    assert set(reg._cache) == {("patchcore", "b"), ("patchcore", "c")}
    reg.get("patchcore", "b")  # refresh -> "c" now LRU
    reg.get("patchcore", "d")  # evicts "c"
    assert set(reg._cache) == {("patchcore", "b"), ("patchcore", "d")}


def test_run_prediction_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = Registry()
    monkeypatch.setattr(reg, "_load", _fake_loaded)
    pred = run_prediction(_png(), "leather", "patchcore", registry=reg)
    assert pred.category == "leather"
    assert pred.model == "patchcore"
    assert pred.verdict == "normal"  # score 0.42 < threshold 0.5
    assert 0.0 <= pred.normalized_score <= 1.0
    assert pred.heatmap_png[:8] == b"\x89PNG\r\n\x1a\n"
    assert pred.mask_png[:8] == b"\x89PNG\r\n\x1a\n"
    assert pred.latency_ms >= 0
