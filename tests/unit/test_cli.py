from __future__ import annotations

import pytest

from mddf import cli
from mddf.data.layout import CategoryStatus, DatasetStatus


def _fake_status(ok: bool = True) -> DatasetStatus:
    issues: list[str] = [] if ok else ["test image count 1 != expected 42"]
    return DatasetStatus(
        root="/tmp/datasets/MVTecAD",
        categories=[
            CategoryStatus(
                name="toothbrush",
                present=True,
                train_good=60,
                test_total=42,
                test_good=12,
                defect_types=["defective"],
                has_ground_truth=True,
                expected_train_good=60,
                expected_test_total=42,
                issues=issues,
            )
        ],
    )


def test_data_command_reports_table(monkeypatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(
        "mddf.data.download.ensure_categories",
        lambda categories, **_: _fake_status(ok=True),
    )
    rc = cli.main(["data", "--category", "toothbrush"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "toothbrush" in out
    assert "ok" in out


def test_data_command_fails_on_bad_dataset(monkeypatch) -> None:
    def _raise(categories: object, **_: object) -> DatasetStatus:
        raise RuntimeError("MVTec AD verification failed for: toothbrush.")

    monkeypatch.setattr("mddf.data.download.ensure_categories", _raise)
    assert cli.main(["data"]) == 1


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert "mddf" in capsys.readouterr().out


def test_train_stub_still_guarded() -> None:
    assert cli.main(["export"]) == 2
