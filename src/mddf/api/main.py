"""FastAPI application factory.

Layout:
  * ``/api/v1/*``  — JSON API (health, categories, benchmark, predict[M6]).
  * ``/metrics``   — Prometheus exposition.
  * ``/``          — the built React SPA from ``web/dist`` when present.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response
from starlette.staticfiles import StaticFiles

from mddf.api.errors import ApiError, install_error_handlers
from mddf.api.middleware import ObservabilityMiddleware
from mddf.api.routes import health, meta, predict
from mddf.config import REPO_ROOT, get_settings, project_version
from mddf.logging import configure_logging, get_logger

API_PREFIX = "/api/v1"
SPA_DIR = REPO_ROOT / "web" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(level=settings.log_level, json=settings.log_json)
    log = get_logger("mddf.startup")
    log.info("starting", version=project_version(), spa=SPA_DIR.is_dir())

    from mddf.inference.registry import get_registry

    try:
        available = get_registry().available()
        log.info("models_available", count=len(available))
        if settings.prefetch_on_startup and available:
            get_registry().warmup(available[: settings.registry_cache_size])
    except Exception:  # startup must not crash on artifact issues
        log.exception("registry_probe_failed")

    yield
    log.info("shutdown")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.api_title,
        version=project_version(),
        summary="Unsupervised visual defect detection (PatchCore + EfficientAD), CPU-only.",
        lifespan=lifespan,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )

    app.add_middleware(ObservabilityMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    install_error_handlers(app)

    app.include_router(health.router, prefix=API_PREFIX)
    app.include_router(meta.router, prefix=API_PREFIX)
    app.include_router(predict.router, prefix=API_PREFIX)

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/api", include_in_schema=False)
    async def api_index() -> JSONResponse:
        return JSONResponse(
            {
                "service": settings.api_title,
                "version": project_version(),
                "docs": "/api/docs",
                "endpoints": [
                    f"{API_PREFIX}/health",
                    f"{API_PREFIX}/categories",
                    f"{API_PREFIX}/benchmark",
                ],
            }
        )

    # Catch-all so unknown /api/** paths return problem+json even when the SPA is
    # mounted at "/" (StaticFiles would otherwise serve its own plain 404).
    @app.api_route(
        "/api/{rest:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        include_in_schema=False,
    )
    async def api_not_found(rest: str) -> Response:
        raise ApiError(status_code=404, slug="not-found", detail=f"No API route /api/{rest}")

    if SPA_DIR.is_dir():
        app.mount("/", StaticFiles(directory=SPA_DIR, html=True), name="spa")
    else:

        @app.get("/", include_in_schema=False)
        async def root() -> Response:
            return RedirectResponse(url="/api/docs")

    return app


app = create_app()
