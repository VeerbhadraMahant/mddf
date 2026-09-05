from __future__ import annotations

from mddf import __version__
from mddf.config import Settings, get_settings, project_version


def test_settings_singleton() -> None:
    assert get_settings() is get_settings()


def test_defaults_are_cpu_safe() -> None:
    s = Settings()
    assert s.max_upload_bytes <= 16 * 1024 * 1024
    assert s.default_model in ("patchcore", "efficient_ad")
    assert s.default_input_shape == (s.input_size, s.input_size)


def test_env_override(monkeypatch) -> None:
    monkeypatch.setenv("MDDF_INPUT_SIZE", "320")
    monkeypatch.setenv("MDDF_DEFAULT_MODEL", "efficient_ad")
    s = Settings()
    assert s.input_size == 320
    assert s.default_model == "efficient_ad"


def test_version_matches_package() -> None:
    assert project_version() == __version__
