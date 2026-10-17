"""FastAPI application factory for the Sprint 2 operational backend."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.app.api.router import api_router, health_router
from backend.app.core.config import Settings, get_settings
from backend.app.db.session import AsyncSessionFactory, engine as default_engine
from backend.app.services.inference import InferenceService


_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,120}$")


def create_app(
    *,
    settings: Settings | None = None,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
    inference_service: InferenceService | None = None,
) -> FastAPI:
    cfg = settings or get_settings()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            # An injected factory is owned by its caller (notably the SQLite
            # integration tests).  The process-global production engine is
            # owned by the default application and closed on shutdown.
            if session_factory is None:
                await default_engine.dispose()

    application = FastAPI(
        title=cfg.app_name,
        version="2.0.0",
        description=(
            "Leakage-safe transaction scoring and capacity-aware AML analyst workflow. "
            "Evaluation ground truth is intentionally absent from every API contract."
        ),
        lifespan=lifespan,
    )
    application.state.settings = cfg
    application.state.session_factory = session_factory or AsyncSessionFactory

    if inference_service is None:
        inference_service, load_error = _load_configured_model(cfg.model_artifact_dir)
        application.state.model_load_error = load_error
    else:
        application.state.model_load_error = None
    application.state.inference_service = inference_service

    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(cfg.cors_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID"],
    )

    @application.middleware("http")
    async def attach_request_id(request: Any, call_next: Any) -> Any:
        supplied = request.headers.get("X-Request-ID", "")
        request_id = supplied if _REQUEST_ID.fullmatch(supplied) else uuid4().hex
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @application.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {"service": cfg.app_name, "api": "/api/v1", "health": "/health/live"}

    application.include_router(health_router)
    application.include_router(api_router)
    return application


def _load_configured_model(root: Path) -> tuple[InferenceService, str | None]:
    candidates: list[Path] = []
    configured = Path(root)
    if (configured / "manifest.json").is_file():
        candidates.append(configured)
    if configured.is_dir():
        candidates.extend(
            path.parent
            for path in configured.glob("*/manifest.json")
            if path.parent not in candidates
        )
    if not candidates:
        return InferenceService(), f"no model bundle found under {configured}"
    candidates.sort(key=lambda item: (item.stat().st_mtime_ns, item.name), reverse=True)
    errors: list[str] = []
    for candidate in candidates:
        try:
            return InferenceService.from_path(candidate), None
        except Exception as exc:  # Startup remains available for health/debugging.
            errors.append(f"{candidate.name}: {exc}")
    return InferenceService(), "; ".join(errors)


app = create_app()


__all__ = ["app", "create_app"]
