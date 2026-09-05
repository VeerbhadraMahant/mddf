"""Tests for the dependency-free MVTec AD layout inspector."""

from __future__ import annotations

from pathlib import Path

import pytest

from mddf.catalog import load_catalog
from mddf.data.layout import find_category_dir, inspect_category, inspect_dataset

# Smallest MVTec AD category — keeps the synthetic tree tiny.
CATEGORY = "toothbrush"


def _touch_pngs(directory: Path, count: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (directory / f"{i:03d}.png").write_bytes(b"")


@pytest.fixture
def good_tree(tmp_path: Path) -> Path:
    """A structurally correct MVTec AD tree for one category, matching catalog counts."""
    entry = load_catalog().get(CATEGORY)
    root = tmp_path / "MVTecAD"
    cdir = root / CATEGORY
    _touch_pngs(cdir / "train" / "good", entry.train_good)

    test_good = 12
    defect_a = 15
    defect_b = entry.test_total - test_good - defect_a
    _touch_pngs(cdir / "test" / "good", test_good)
    _touch_pngs(cdir / "test" / "defect_a", defect_a)
    _touch_pngs(cdir / "test" / "defect_b", defect_b)
    _touch_pngs(cdir / "ground_truth" / "defect_a", defect_a)
    _touch_pngs(cdir / "ground_truth" / "defect_b", defect_b)
    return root


def test_find_category_dir_handles_wrapper(good_tree: Path) -> None:
    assert find_category_dir(good_tree, CATEGORY) == good_tree / CATEGORY
    # Also when root points one level up.
    assert find_category_dir(good_tree.parent, CATEGORY) == good_tree / CATEGORY


def test_correct_tree_passes(good_tree: Path) -> None:
    status = inspect_category(good_tree, CATEGORY)
    assert status.present
    assert status.ok, status.issues
    assert status.defect_types == ["defect_a", "defect_b"]
    assert status.has_ground_truth


def test_missing_category_is_reported(tmp_path: Path) -> None:
    status = inspect_category(tmp_path, CATEGORY)
    assert not status.present
    assert not status.ok
    assert "not found" in status.issues[0]


def test_wrong_counts_flagged(good_tree: Path) -> None:
    next((good_tree / CATEGORY / "train" / "good").glob("*.png")).unlink()
    status = inspect_category(good_tree, CATEGORY)
    assert not status.ok
    assert any("train/good count" in i for i in status.issues)


def test_missing_ground_truth_flagged(good_tree: Path) -> None:
    for p in (good_tree / CATEGORY / "ground_truth").rglob("*.png"):
        p.unlink()
    status = inspect_category(good_tree, CATEGORY)
    assert any("ground_truth" in i for i in status.issues)


def test_inspect_dataset_subset(good_tree: Path) -> None:
    status = inspect_dataset(good_tree, [CATEGORY])
    assert status.ok
    assert status.missing == []
    assert [c.name for c in status.categories] == [CATEGORY]
