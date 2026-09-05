"""Lazy, LRU-cached loader for exported model artifacts.

For a given ``(model, category)`` it resolves three files — ``model.onnx``,
``preprocess.json``, ``metrics.json`` — preferring the local ``artifacts/`` tree
and falling back to a Hugging Face *model* repo (pulled once, then disk-cached by
``huggingface_hub``). Built :class:`LoadedModel` objects are kept in a small
in-RAM LRU so a busy category stays warm and cold ones don't hold memory.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from mddf.config import ALL_MODELS, ModelName, Settings, get_settings
from mddf.inference.onnx_session import OnnxModel
from mddf.inference.threshold import ThresholdModel
from mddf.inference.threshold import build as build_threshold
from mddf.logging import get_logger
from mddf.training import artifacts as art
from mddf.training.preprocess_spec import PreprocessSpec

_log = get_logger("mddf.inference.registry")

_FILES = ("model.onnx", "preprocess.json", "metrics.json")
_INT8_NAME = "model.int8.onnx"


class ArtifactsUnavailable(RuntimeError):
    """Raised when a category's files are neither local nor fetchable."""


@dataclass
class LoadedModel:
    model: ModelName
    category: str
    session: OnnxModel
    spec: PreprocessSpec
    threshold: ThresholdModel
    metrics: dict[str, float]
    version: str


def _hf_fetch(repo: str, revision: str, rel: str, cache_root: Path) -> Path | None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:  # pragma: no cover
        return None
    try:
        got = hf_hub_download(
            repo_id=repo,
            filename=rel,
            revision=revision,
            repo_type="model",
            local_dir=cache_root,
        )
        return Path(got)
    except Exception as exc:  # network / 404 / auth
        _log.warning("hf_fetch_failed", rel=rel, error=str(exc))
        return None


def _resolve_one(
    model: ModelName, category: str, name: str, settings: Settings, *, required: bool = True
) -> Path | None:
    local = art.category_dir(model, category, root=settings.artifacts_dir) / name
    if local.is_file():
        return local
    if settings.hf_model_repo:
        fetched = _hf_fetch(
            settings.hf_model_repo,
            settings.hf_revision,
            f"{model}/{category}/{name}",
            settings.artifacts_dir,
        )
        if fetched and fetched.is_file():
            return fetched
    if required:
        raise ArtifactsUnavailable(f"{model}/{category}: cannot resolve {name}")
    return None


def _resolve_files(model: ModelName, category: str, settings: Settings) -> dict[str, Path]:
    resolved: dict[str, Path] = {
        name: p
        for name in ("preprocess.json", "metrics.json")
        if (p := _resolve_one(model, category, name, settings)) is not None
    }
    onnx: Path | None = None
    if settings.prefer_int8:
        onnx = _resolve_one(model, category, _INT8_NAME, settings, required=False)
        if onnx is not None:
            _log.info("using_int8", model=model, category=category)
    if onnx is None:
        onnx = _resolve_one(model, category, "model.onnx", settings)
    assert onnx is not None  # _resolve_one(required=True) raises otherwise
    resolved["model.onnx"] = onnx
    return resolved


def _threshold_from_metrics(metrics_doc: dict[str, object]) -> ThresholdModel:
    thresholds = metrics_doc.get("thresholds") or {}
    raw = metrics_doc.get("raw_metrics") or {}
    thr = None
    if isinstance(thresholds, dict):
        thr = thresholds.get("image")
    ref = None
    if isinstance(raw, dict):
        lo, hi = raw.get("min_image_score"), raw.get("max_image_score")
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
            ref = (float(lo), float(hi))
    return build_threshold(float(thr) if isinstance(thr, (int, float)) else None, ref_scores=ref)


class Registry:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._cache: OrderedDict[tuple[str, str], LoadedModel] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return max(self._settings.registry_cache_size, 1)

    def _load(self, model: ModelName, category: str) -> LoadedModel:
        files = _resolve_files(model, category, self._settings)
        spec = PreprocessSpec.model_validate(art.read_json(files["preprocess.json"]))
        metrics_doc = art.read_json(files["metrics.json"])
        metrics = {
            k: float(v)
            for k, v in (metrics_doc.get("metrics") or {}).items()
            if isinstance(v, (int, float))
        }
        session = OnnxModel(files["model.onnx"], threads=1)
        version = str(metrics_doc.get("git_sha") or metrics_doc.get("exported_at") or "dev")
        _log.info("model_loaded", model=model, category=category, outputs=session.output_names)
        return LoadedModel(
            model=model,
            category=category,
            session=session,
            spec=spec,
            threshold=_threshold_from_metrics(metrics_doc),
            metrics=metrics,
            version=version,
        )

    def get(self, model: ModelName, category: str) -> LoadedModel:
        key = (model, category)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        loaded = self._load(model, category)  # outside the lock: I/O heavy
        with self._lock:
            self._cache[key] = loaded
            self._cache.move_to_end(key)
            while len(self._cache) > self.capacity:
                evicted, _ = self._cache.popitem(last=False)
                _log.info("model_evicted", model=evicted[0], category=evicted[1])
        return loaded

    def available(self) -> list[tuple[ModelName, str]]:
        from mddf.catalog import load_catalog

        out: list[tuple[ModelName, str]] = []
        for model in ALL_MODELS:
            for cat in load_catalog().names:
                if art.onnx_path(model, cat, root=self._settings.artifacts_dir).is_file():
                    out.append((model, cat))
        return out

    def warmup(self, pairs: list[tuple[ModelName, str]]) -> None:
        for model, category in pairs:
            try:
                self.get(model, category)
            except ArtifactsUnavailable as exc:
                _log.warning("warmup_skip", model=model, category=category, error=str(exc))


_registry: Registry | None = None


def get_registry() -> Registry:
    global _registry
    if _registry is None:
        _registry = Registry()
    return _registry


__all__ = ["ArtifactsUnavailable", "LoadedModel", "Registry", "get_registry"]
