"""Request-id + timing + access-log middleware, and Prometheus instrumentation."""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

import structlog
from prometheus_client import Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from mddf.logging import get_logger

REQUEST_ID_HEADER = "x-request-id"

_REQUESTS = Counter("mddf_http_requests_total", "HTTP requests", ["method", "path", "status"])
_LATENCY = Histogram(
    "mddf_http_request_duration_seconds", "HTTP request duration", ["method", "path"]
)
_log = get_logger("mddf.access")


class ObservabilityMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        request.state.request_id = request_id
        structlog.contextvars.bind_contextvars(request_id=request_id)

        # Route template ("/api/v1/predict") not the concrete path, to keep label
        # cardinality bounded.
        route = request.scope.get("route")
        path_label = getattr(route, "path", request.url.path)

        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            elapsed = time.perf_counter() - start
            _log.exception(
                "request_failed",
                method=request.method,
                path=request.url.path,
                elapsed_ms=round(elapsed * 1000, 2),
            )
            _REQUESTS.labels(request.method, path_label, 500).inc()
            _LATENCY.labels(request.method, path_label).observe(elapsed)
            raise
        finally:
            structlog.contextvars.unbind_contextvars("request_id")

        elapsed = time.perf_counter() - start
        response.headers[REQUEST_ID_HEADER] = request_id
        _REQUESTS.labels(request.method, path_label, response.status_code).inc()
        _LATENCY.labels(request.method, path_label).observe(elapsed)
        _log.info(
            "request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            elapsed_ms=round(elapsed * 1000, 2),
        )
        return response


__all__ = ["REQUEST_ID_HEADER", "ObservabilityMiddleware"]
