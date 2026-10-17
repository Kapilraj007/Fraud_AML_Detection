"""Administrator-only access to sanitized operational audit events."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from backend.app.api.dependencies import DatabaseSession, require_roles
from backend.app.db.enums import UserRole
from backend.app.db.models import AuditEvent, User


router = APIRouter(prefix="/audit", tags=["audit"])
AdminUser = Annotated[User, Depends(require_roles(UserRole.ADMIN))]


class AuditEventRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    actor_user_id: UUID | None
    event_type: str
    resource_type: str
    resource_id: str | None
    request_id: str | None
    safe_details: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


@router.get("/events", response_model=list[AuditEventRead])
async def list_audit_events(
    session: DatabaseSession,
    user: AdminUser,
    event_type: str | None = None,
    resource_type: str | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0, le=1_000_000)] = 0,
) -> list[AuditEvent]:
    statement = select(AuditEvent)
    if event_type:
        statement = statement.where(AuditEvent.event_type == event_type.strip())
    if resource_type:
        statement = statement.where(AuditEvent.resource_type == resource_type.strip())
    result = await session.execute(
        statement.order_by(AuditEvent.created_at.desc(), AuditEvent.id.desc())
        .offset(offset)
        .limit(limit)
    )
    return list(result.scalars())


__all__ = ["router"]
