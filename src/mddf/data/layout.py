"""Pure filesystem inspection of an MVTec AD tree — no torch, no network.

MVTec AD on disk::

    <root>/<category>/
        train/good/*.png
        test/good/*.png
        test/<defect_type>/*.png
        ground_truth/<defect_type>/*_mask.png

This module verifies that layout and cross-checks the image counts against the
published numbers in ``configs/categories.yaml``. Kept dependency-free so it can be
unit-tested against a synthetic directory tree.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from mddf.catalog import load_catalog

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


class CategoryStatus(BaseModel):
    name: str
    present: bool
    train_good: int
    test_total: int
    test_good: int
    defect_types: list[str]
    has_ground_truth: bool
    expected_train_good: int
    expected_test_total: int
    issues: list[str]

    @property
    def ok(self) -> bool:
        return self.present and not self.issues


class DatasetStatus(BaseModel):
    root: str
    categories: list[CategoryStatus]

    @property
    def ok(self) -> bool:
        return bool(self.categories) and all(c.ok for c in self.categories)

    @property
    def missing(self) -> list[str]:
        return [c.name for c in self.categories if not c.present]


def _count_images(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for p in directory.rglob("*") if p.suffix.lower() in _IMAGE_SUFFIXES)


def find_category_dir(root: Path, category: str) -> Path | None:
    """Locate ``<category>/`` under ``root``, tolerating a wrapper dir.

    Anomalib extracts to e.g. ``datasets/MVTecAD/<category>`` or
    ``datasets/mvtec_anomaly_detection/<category>`` depending on version.
    """
    candidates = (
        [root / category, *(d / category for d in root.iterdir() if d.is_dir())]
        if root.is_dir()
        else []
    )
    for c in candidates:
        if (c / "train" / "good").is_dir():
            return c
    return None


def inspect_category(root: Path, category: str) -> CategoryStatus:
    catalog_entry = load_catalog().get(category)
    cdir = find_category_dir(root, category)

    if cdir is None:
        return CategoryStatus(
            name=category,
            present=False,
            train_good=0,
            test_total=0,
            test_good=0,
            defect_types=[],
            has_ground_truth=False,
            expected_train_good=catalog_entry.train_good,
            expected_test_total=catalog_entry.test_total,
            issues=["category directory not found"],
        )

    train_good = _count_images(cdir / "train" / "good")
    test_dir = cdir / "test"
    defect_types = (
        sorted(d.name for d in test_dir.iterdir() if d.is_dir() and d.name != "good")
        if test_dir.is_dir()
        else []
    )
    test_good = _count_images(test_dir / "good")
    test_total = _count_images(test_dir)
    has_gt = (cdir / "ground_truth").is_dir() and _count_images(cdir / "ground_truth") > 0

    issues: list[str] = []
    if train_good == 0:
        issues.append("no defect-free training images")
    if train_good != catalog_entry.train_good:
        issues.append(f"train/good count {train_good} != expected {catalog_entry.train_good}")
    if test_total != catalog_entry.test_total:
        issues.append(f"test image count {test_total} != expected {catalog_entry.test_total}")
    if not defect_types:
        issues.append("no defect classes under test/")
    if not has_gt:
        issues.append("missing ground_truth masks")

    return CategoryStatus(
        name=category,
        present=True,
        train_good=train_good,
        test_total=test_total,
        test_good=test_good,
        defect_types=defect_types,
        has_ground_truth=has_gt,
        expected_train_good=catalog_entry.train_good,
        expected_test_total=catalog_entry.test_total,
        issues=issues,
    )


def inspect_dataset(root: Path, categories: list[str] | None = None) -> DatasetStatus:
    names = categories or load_catalog().names
    return DatasetStatus(
        root=str(root),
        categories=[inspect_category(root, name) for name in names],
    )


__all__ = [
    "CategoryStatus",
    "DatasetStatus",
    "find_category_dir",
    "inspect_category",
    "inspect_dataset",
]
