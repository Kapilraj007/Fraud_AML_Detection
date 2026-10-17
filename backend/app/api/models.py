"""Model registry read and controlled promotion routes."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status

from backend.app.api.dependencies import CurrentUser, DatabaseSession, audit_request, require_roles
from backend.app.db.enums import ModelStatus, UserRole
from backend.app.db.models import ModelVersion, User
from backend.app.db.repositories import get_active_model_version, promote_model_version
from backend.app.schemas.model import ModelPromotionRequest, ModelVersionRead


router = APIRouter(prefix="/models", tags=["models"])
AdminUser = Annotated[User, Depends(require_roles(UserRole.ADMIN))]


@router.get("/active", response_model=ModelVersionRead)
async def active_model(
    session: DatabaseSession,
    user: CurrentUser,
) -> ModelVersion:
    model = await get_active_model_version(session)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="no active model is registered",
        )
    return model


@router.post("/{model_version_id}/promote", response_model=ModelVersionRead)
async def promote_model(
    model_version_id: UUID,
    payload: ModelPromotionRequest,
    request: Request,
    session: DatabaseSession,
    user: AdminUser,
) -> ModelVersion:
    current = await get_active_model_version(session)
    if payload.expected_current_version is not None:
        current_version = None if current is None else current.version
        if current_version != payload.expected_current_version:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="active model changed; reload registry state before promoting",
            )
    target = await session.get(ModelVersion, model_version_id)
    if target is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="model not found")
    if target.status == ModelStatus.FAILED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="a failed model cannot be promoted",
        )
    promoted = await promote_model_version(session, model_version_id=model_version_id)
    await audit_request(
        session,
        request,
        event_type="model.promoted",
        resource_type="model_version",
        actor=user,
        resource_id=promoted.id,
        details={
            "previous_model_id": None if current is None else str(current.id),
            "reason": payload.reason.strip(),
        },
    )
    await session.commit()
    await session.refresh(promoted)
    return promoted


__all__ = ["router"]
