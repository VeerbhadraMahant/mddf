from __future__ import annotations

import pytest

from mddf.catalog import load_catalog

MVTEC_AD_CATEGORIES = {
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
}


def test_catalog_has_all_15_mvtec_categories() -> None:
    catalog = load_catalog()
    assert set(catalog.names) == MVTEC_AD_CATEGORIES
    assert catalog.dataset == "mvtec_ad"


def test_every_category_has_a_published_baseline() -> None:
    for category in load_catalog().categories:
        assert 0.9 <= category.published_image_auroc <= 1.0
        assert category.kind in ("object", "texture")
        assert category.train_good > 0
        assert category.test_total > 0


def test_lookup_and_missing() -> None:
    catalog = load_catalog()
    assert catalog.get("leather").kind == "texture"
    with pytest.raises(KeyError):
        catalog.get("nonexistent")
