"""Centralised configuration.

Runtime settings come from environment variables (prefix ``MDDF_``) or a local
``.env`` file. The dataset/model catalogue is read from ``configs/categories.yaml``
so training and serving agree on the exact set of categories and their metadata.
"""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"

ModelName = Literal["patchcore", "efficient_ad"]


class Settings(BaseSettings):
    """Process-wide runtime settings."""

    model_config = SettingsConfigDict(
        env_prefix="MDDF_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Service ---
    api_title: str = "Manufacturing Defect Detection System"
    api_version: str = "0.1.0"
    log_level: str = "INFO"
    log_json: bool = True

    # --- Upload guards ---
    max_upload_bytes: int = 8 * 1024 * 1024
    max_image_pixels: int = 4096 * 4096
    allowed_content_types: tuple[str, ...] = ("image/jpeg", "image/png", "image/bmp", "image/webp")

    # --- Inference ---
    default_model: ModelName = "patchcore"
    input_size: int = 256
    registry_cache_size: int = 4  # model artifact sets kept warm in RAM

    # --- Artifacts ---
    artifacts_dir: Path = REPO_ROOT / "artifacts"
    hf_model_repo: str = "veerbhadra/mddf-artifacts"
    hf_revision: str = "main"
    prefetch_on_startup: bool = False

    @property
    def default_input_shape(self) -> tuple[int, int]:
        return (self.input_size, self.input_size)


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()


@lru_cache
def project_version() -> str:
    """Read the package version from pyproject.toml (single source of truth)."""
    pyproject = REPO_ROOT / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    version = data["project"]["version"]
    assert isinstance(version, str)
    return version


# Re-exported for convenience.
settings = get_settings()

__all__ = [
    "CONFIGS_DIR",
    "REPO_ROOT",
    "ModelName",
    "Settings",
    "get_settings",
    "project_version",
    "settings",
]
