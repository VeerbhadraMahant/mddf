"""Centralised configuration.

Runtime settings come from environment variables (prefix ``MDDF_``) or a local
``.env`` file. The dataset/model catalogue and per-model configs ship inside the
package (``mddf/resources/``) so training and serving — checkout or wheel — agree
on the exact set of categories and hyper-parameters.
"""

from __future__ import annotations

import tomllib
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
# Bundled with the package so it resolves the same in a checkout and in a wheel.
CONFIGS_DIR = Path(__file__).resolve().parent / "resources"

ModelName = Literal["padim", "patchcore", "efficient_ad"]
# Ordered weakest→strongest baseline for tables: distribution / memory-bank / distillation.
ALL_MODELS: tuple[ModelName, ...] = ("padim", "patchcore", "efficient_ad")


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

    # --- Data (training only) ---
    dataset_root: Path = REPO_ROOT / "datasets"

    # --- Artifacts ---
    artifacts_dir: Path = REPO_ROOT / "artifacts"
    hf_model_repo: str = "bhadra244131/mddf-artifacts"
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
    """Package version: installed metadata first, else pyproject.toml in a checkout."""
    try:
        return version("mddf")
    except PackageNotFoundError:
        pass
    pyproject = REPO_ROOT / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    v = data["project"]["version"]
    assert isinstance(v, str)
    return v


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
