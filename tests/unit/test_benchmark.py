from __future__ import annotations

from pathlib import Path

import pytest

from mddf.benchmark.accuracy import aggregate, comparison_markdown
from mddf.training import artifacts as art


def _write_metrics(root: Path, model: str, category: str, image_auroc: float) -> None:
    art.write_json(
        art.metrics_path(model, category, root=root),  # type: ignore[arg-type]
        {
            "model": model,
            "category": category,
            "metrics": {"image_auroc": image_auroc, "pixel_auroc": 0.97, "aupro": 0.94},
            "thresholds": {"image": 11.5},
        },
    )


def test_aggregate_builds_reporting_shape(tmp_path: Path) -> None:
    _write_metrics(tmp_path, "patchcore", "leather", 0.995)
    _write_metrics(tmp_path, "efficient_ad", "leather", 0.972)
    _write_metrics(tmp_path, "patchcore", "grid", 0.981)

    doc = aggregate(["patchcore", "efficient_ad"], root=tmp_path)
    assert set(doc["results"]) == {"leather", "grid"}
    assert doc["results"]["leather"]["patchcore"]["image_auroc"] == 0.995
    assert doc["results"]["leather"]["patchcore"]["threshold_image"] == 11.5
    assert "efficient_ad" not in doc["results"]["grid"]
    # written to disk
    assert (tmp_path / "benchmark" / "metrics.json").is_file()


def test_aggregate_empty_when_nothing_trained(tmp_path: Path) -> None:
    assert aggregate(["patchcore"], root=tmp_path, write=False)["results"] == {}


def test_comparison_markdown_has_delta_column(tmp_path: Path) -> None:
    _write_metrics(tmp_path, "patchcore", "leather", 0.995)
    md = comparison_markdown(root=tmp_path)
    assert "Published (PatchCore)" in md
    assert "| leather | patchcore |" in md
    assert "+" in md or "-" in md  # signed delta rendered


@pytest.mark.slow
def test_measure_onnx_on_a_tiny_model(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from mddf.benchmark.latency import measure_onnx

    model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.AdaptiveAvgPool2d(1))
    model.eval()
    onnx_file = tmp_path / "tiny.onnx"
    torch.onnx.export(
        model,
        torch.randn(1, 3, 32, 32),
        str(onnx_file),
        input_names=["input"],
        dynamo=False,
    )
    stats = measure_onnx(onnx_file, (1, 3, 32, 32), runs=5, warmup=2)
    assert stats["latency_ms_p50"] >= 0
    assert stats["runs"] == 5
