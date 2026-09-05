from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _isolate_artifacts(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch):
    """Every test sees an empty artifacts/ and dataset/ dir, regardless of what a
    real training run left in the working tree, and fresh settings/registry
    singletons."""
    import mddf.config as config
    import mddf.inference.registry as registry
    import mddf.reporting as reporting

    art_dir = tmp_path_factory.mktemp("artifacts")
    ds_dir = tmp_path_factory.mktemp("datasets")
    monkeypatch.setenv("MDDF_ARTIFACTS_DIR", str(art_dir))
    monkeypatch.setenv("MDDF_DATASET_ROOT", str(ds_dir))
    config.get_settings.cache_clear()
    monkeypatch.setattr(config, "settings", config.get_settings())
    monkeypatch.setattr(registry, "_registry", None)
    reporting.clear_cache()
    yield
    config.get_settings.cache_clear()
    reporting.clear_cache()


@pytest.fixture
def client() -> Iterator[TestClient]:
    from mddf.api.main import create_app

    with TestClient(create_app()) as c:
        yield c
