"""The FastAPI app: routes, the error body from docs/API.md, and the built dashboard."""

from __future__ import annotations

import logging
import socket
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException

from voicelab.api.guards import (
    MAX_JSON_BYTES,
    BodyLimit,
    SameSiteOnly,
    error_response,
    trusted_sites,
)
from voicelab.api.routes import calls, library, live, models, runs
from voicelab.services.container import Services, build_services
from voicelab.services.errors import NotFound, ServiceError
from voicelab.services.system import lan_ip
from voicelab.store.calls import BadCursor

logger = logging.getLogger(__name__)

_HTTP_CODES = {404: "not_found", 405: "invalid", 409: "conflict", 413: "invalid"}
_error = error_response


def _install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ServiceError)
    async def service_error(_: Request, exc: ServiceError) -> JSONResponse:
        return _error(exc.status, exc.code, exc.message, exc.detail)

    @app.exception_handler(BadCursor)
    async def bad_cursor(_: Request, __: BadCursor) -> JSONResponse:
        return _error(400, "invalid", "bad cursor")

    @app.exception_handler(RequestValidationError)
    async def bad_request(_: Request, exc: RequestValidationError) -> JSONResponse:
        problems = [
            {"field": ".".join(str(part) for part in e["loc"]), "problem": e["msg"]}
            for e in exc.errors()
        ]
        return _error(400, "invalid", "invalid request", {"problems": problems})

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException) -> JSONResponse:
        return _error(
            exc.status_code, _HTTP_CODES.get(exc.status_code, "internal"), str(exc.detail)
        )

    @app.exception_handler(Exception)
    async def unexpected(_: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled error", exc_info=exc)
        return _error(500, "internal", f"{type(exc).__name__}: {exc}")


def _mount_dashboard(app: FastAPI, dist: Path) -> None:
    root = dist.resolve()

    @app.get("/{path:path}", include_in_schema=False)
    async def dashboard(path: str) -> FileResponse:
        if path == "api" or path.startswith(("api/", "ws/")):
            raise NotFound(f"no route {path}")
        target = (root / path).resolve()
        if not target.is_relative_to(root):
            raise NotFound("outside the dashboard")
        if path and target.is_file():
            return FileResponse(target)
        index = root / "index.html"
        if not index.exists():
            raise NotFound("the dashboard isn't built: run `pnpm build` in web/")
        return FileResponse(index)  # React Router owns every other path


def create_app(services: Services | None = None, *, seed: bool = True) -> FastAPI:
    services = services or build_services()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await services.start(seed=seed)
        yield
        await services.stop()

    app = FastAPI(
        title="voicelab", lifespan=lifespan, docs_url="/api/docs", openapi_url="/api/openapi.json"
    )
    app.state.services = services
    settings = services.settings

    def body_limit(path: str) -> int:
        return settings.max_upload_bytes if path == "/api/uploads" else MAX_JSON_BYTES

    # The last one added runs first, so a refused site never gets as far as being read.
    app.add_middleware(BodyLimit, limit_for=body_limit)
    app.add_middleware(SameSiteOnly, sites=trusted_sites(settings, lan_ip(), socket.gethostname()))
    _install_error_handlers(app)
    for module in (calls, library, runs, live, models):
        app.include_router(module.router)
    _mount_dashboard(app, services.settings.web_dist)
    return app
