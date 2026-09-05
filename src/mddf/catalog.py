"""MVTec AD category catalogue loaded from ``mddf/resources/categories.yaml``.

Both the training pipeline and the inference service import from here so they can
never disagree about which categories exist or what their published baselines are.
"""

from __future__ import annotations

from functools import lru_cache

import yaml
from pydantic import BaseModel, Field

from mddf.config import CONFIGS_DIR

CATEGORIES_FILE = CONFIGS_DIR / "categories.yaml"


class Category(BaseModel):
    """One MVTec AD category and its published PatchCore baseline."""

    name: str
    kind: str = Field(pattern="^(object|texture)$")
    train_good: int
    test_total: int
    published_image_auroc: float


class Catalog(BaseModel):
    dataset: str
    categories: list[Category]

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.categories]

    def get(self, name: str) -> Category:
        for category in self.categories:
            if category.name == name:
                return category
        raise KeyError(name)


@lru_cache
def load_catalog() -> Catalog:
    raw = yaml.safe_load(CATEGORIES_FILE.read_text(encoding="utf-8"))
    return Catalog.model_validate(raw)


__all__ = ["CATEGORIES_FILE", "Catalog", "Category", "load_catalog"]
