"""Liveness and readiness probes."""

from __future__ import annotations

from fastapi import APIRouter

from mddf.api.schemas import HealthResponse, ReadyResponse
from mddf.config import get_settings, project_version

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness: the process is up and serving."""
    settings = get_settings()
    return HealthResponse(service=settings.api_title, version=project_version())


@router.get("/ready", response_model=ReadyResponse)
async def ready() -> ReadyResponse:
    """Readiness: model artifacts are reachable and at least one category loads.

    Filled in at milestone M6 once the inference registry exists; for now the
    service is ready as soon as it is live.
    """
    return ReadyResponse(ready=True, detail="Service is live. Model registry check added in M6.")
