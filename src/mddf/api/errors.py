"""Structured JSON error handling (RFC 9457 "problem details" shape).

Every failure the client can cause is raised as :class:`ApiError` and rendered as a
JSON body with a stable ``type`` slug — never an HTML page or a stack trace.
"""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

PROBLEM_CONTENT_TYPE = "application/problem+json"


class ProblemDetail(BaseModel):
    type: str
    title: str
    status: int
    detail: str
    request_id: str | None = None


class ApiError(Exception):
    """Raise for any client-caused failure."""

    def __init__(
        self,
        *,
        status_code: int,
        slug: str,
        detail: str,
    ) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.slug = slug
        self.detail = detail

    def to_problem(self, request_id: str | None) -> ProblemDetail:
        return ProblemDetail(
            type=f"https://mddf.dev/errors/{self.slug}",
            title=self.slug.replace("-", " ").title(),
            status=self.status_code,
            detail=self.detail,
            request_id=request_id,
        )


# --- Convenience constructors -------------------------------------------------


def unknown_category(name: str, known: list[str]) -> ApiError:
    return ApiError(
        status_code=status.HTTP_404_NOT_FOUND,
        slug="unknown-category",
        detail=f"Category {name!r} is not available. Known categories: {', '.join(known)}.",
    )


def unsupported_media_type(content_type: str | None, allowed: tuple[str, ...]) -> ApiError:
    return ApiError(
        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        slug="unsupported-media-type",
        detail=f"Content type {content_type!r} is not accepted. Allowed: {', '.join(allowed)}.",
    )


def payload_too_large(limit_bytes: int) -> ApiError:
    return ApiError(
        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        slug="payload-too-large",
        detail=f"Uploaded image exceeds the {limit_bytes} byte limit.",
    )


def invalid_image(reason: str) -> ApiError:
    return ApiError(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        slug="invalid-image",
        detail=f"Could not decode the uploaded image: {reason}.",
    )


def artifacts_unavailable(detail: str) -> ApiError:
    return ApiError(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        slug="artifacts-unavailable",
        detail=detail,
    )


# --- Wiring ----------------------------------------------------------------


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _handle_api_error(request: Request, exc: ApiError) -> JSONResponse:
        problem = exc.to_problem(_request_id(request))
        return JSONResponse(
            status_code=exc.status_code,
            content=problem.model_dump(),
            media_type=PROBLEM_CONTENT_TYPE,
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        problem = ProblemDetail(
            type="https://mddf.dev/errors/validation-error",
            title="Validation Error",
            status=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="; ".join(
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
            )
            or "Request failed schema validation.",
            request_id=_request_id(request),
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=problem.model_dump(),
            media_type=PROBLEM_CONTENT_TYPE,
        )

    @app.exception_handler(StarletteHTTPException)
    async def _handle_http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        problem = ProblemDetail(
            type="https://mddf.dev/errors/http-error",
            title="HTTP Error",
            status=exc.status_code,
            detail=str(exc.detail),
            request_id=_request_id(request),
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=problem.model_dump(),
            media_type=PROBLEM_CONTENT_TYPE,
        )


__all__ = [
    "ApiError",
    "ProblemDetail",
    "artifacts_unavailable",
    "install_error_handlers",
    "invalid_image",
    "payload_too_large",
    "unknown_category",
    "unsupported_media_type",
]
