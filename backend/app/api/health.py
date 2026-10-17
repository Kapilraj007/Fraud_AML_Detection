"""Process and dependency health probes."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response, status

from backend.app.api.dependencies import DatabaseSession
from backend.app.db.repositories import get_active_model_version
from backend.app.db.session import database_is_ready


router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live")
async def liveness() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
async def readiness(
    request: Request,
    response: Response,
    session: DatabaseSession,
) -> dict[str, object]:
    database_ready = await database_is_ready(session)
    active_model = None
    if database_ready:
        try:
            active_model = await get_active_model_version(session)
        except Exception:
            database_ready = False
    inference = getattr(request.app.state, "inference_service", None)
    model_ready = bool(inference is not None and inference.available)
    registry_ready = active_model is not None
    ready = database_ready and model_ready and registry_ready
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "status": "ready" if ready else "not_ready",
        "database": "ready" if database_ready else "unavailable",
        "model": "ready" if model_ready else "unavailable",
        "active_model_registry": "ready" if registry_ready else "unavailable",
    }


__all__ = ["router"]
