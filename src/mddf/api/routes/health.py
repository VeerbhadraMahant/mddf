"""Liveness and readiness probes."""

from __future__ import annotations

from fastapi import APIRouter

from mddf.api.schemas import HealthResponse, ReadyResponse
from mddf.config import get_settings, project_version
from mddf.inference.registry import get_registry

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness: the process is up and serving."""
    settings = get_settings()
    return HealthResponse(service=settings.api_title, version=project_version())


@router.get("/ready", response_model=ReadyResponse)
async def ready() -> ReadyResponse:
    """Readiness: at least one exported (model, category) is resolvable."""
    try:
        pairs = get_registry().available()
    except Exception as exc:  # readiness must never raise
        return ReadyResponse(ready=False, detail=f"registry error: {exc}")
    if not pairs:
        return ReadyResponse(ready=False, detail="no exported models found")
    return ReadyResponse(ready=True, detail=f"{len(pairs)} (model, category) pairs available")
