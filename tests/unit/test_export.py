from __future__ import annotations

from pathlib import Path

import pytest

from mddf.training.export import export_category, export_matrix


def test_export_without_checkpoint_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="train first"):
        export_category("patchcore", "leather", output_root=tmp_path)


def test_export_matrix_collects_failures(tmp_path: Path) -> None:
    results, failures = export_matrix(
        ["patchcore", "efficient_ad"], ["leather"], output_root=tmp_path
    )
    assert results == []
    got = {(m, c) for m, c, _ in failures}
    assert got == {("patchcore", "leather"), ("efficient_ad", "leather")}
    assert all("FileNotFoundError" in err for *_, err in failures)


@pytest.mark.slow
def test_onnx_signature_reads_io(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from mddf.training.export import _onnx_signature

    model = torch.nn.Conv2d(3, 2, 3, padding=1).eval()
    onnx_file = tmp_path / "m.onnx"
    torch.onnx.export(
        model, torch.randn(1, 3, 16, 16), str(onnx_file), input_names=["x"], dynamo=False
    )
    sig = _onnx_signature(onnx_file, (16, 16))
    assert sig["input_name"] == "x"
    assert sig["output_shapes"][0][:2] == [1, 2]
