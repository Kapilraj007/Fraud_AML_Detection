"""Drift history and human-gated retraining requests."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from backend.app.api.dependencies import CurrentUser, DatabaseSession, audit_request, require_roles
from backend.app.db.enums import UserRole
from backend.app.db.models import DriftSnapshot, RetrainingRequest, User
from backend.app.db.repositories import get_active_model_version
from backend.app.schemas.monitoring import (
    DriftSnapshotRead,
    MonitoringStatusRead,
    RetrainingRequestCreate,
    RetrainingRequestRead,
)


router = APIRouter(prefix="/monitoring", tags=["monitoring"])
RetrainingUser = Annotated[
    User,
    Depends(require_roles(UserRole.SENIOR_INVESTIGATOR, UserRole.ADMIN)),
]


@router.get("/status", response_model=MonitoringStatusRead)
async def monitoring_status(
    session: DatabaseSession,
    user: CurrentUser,
) -> MonitoringStatusRead:
    active = await get_active_model_version(session)
    if active is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="no active model is registered",
        )
    result = await session.execute(
        select(DriftSnapshot)
        .where(DriftSnapshot.model_version_id == active.id)
        .order_by(DriftSnapshot.window_end.desc(), DriftSnapshot.created_at.desc())
        .limit(1)
    )
    latest = result.scalar_one_or_none()
    return MonitoringStatusRead(
        active_model_id=active.id,
        active_model_version=active.version,
        status=None if latest is None else latest.status,
        latest_snapshot=(
            None if latest is None else DriftSnapshotRead.model_validate(latest)
        ),
    )


@router.get("/history", response_model=list[DriftSnapshotRead])
async def monitoring_history(
    session: DatabaseSession,
    user: CurrentUser,
    model_version_id: UUID | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0, le=1_000_000)] = 0,
) -> list[DriftSnapshot]:
    statement = select(DriftSnapshot)
    if model_version_id is not None:
        statement = statement.where(DriftSnapshot.model_version_id == model_version_id)
    result = await session.execute(
        statement.order_by(DriftSnapshot.window_end.desc(), DriftSnapshot.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    return list(result.scalars())


@router.post(
    "/retraining-requests",
    response_model=RetrainingRequestRead,
    status_code=status.HTTP_201_CREATED,
)
@router.post(
    "/retraining/request",
    response_model=RetrainingRequestRead,
    status_code=status.HTTP_201_CREATED,
    include_in_schema=False,
)
async def request_retraining(
    payload: RetrainingRequestCreate,
    request: Request,
    session: DatabaseSession,
    user: RetrainingUser,
) -> RetrainingRequest:
    snapshot = await session.get(DriftSnapshot, payload.drift_snapshot_id)
    if snapshot is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="drift snapshot not found",
        )
    retraining = RetrainingRequest(
        drift_snapshot_id=snapshot.id,
        requested_by_id=user.id,
        reason=payload.reason.strip(),
        before_metrics={
            "status": snapshot.status.value,
            "window_start": snapshot.window_start.isoformat(),
            "window_end": snapshot.window_end.isoformat(),
            "feature_metrics": snapshot.feature_metrics,
            "prediction_metrics": snapshot.prediction_metrics,
            "adwin_detected": snapshot.adwin_detected,
        },
    )
    session.add(retraining)
    await session.flush()
    await audit_request(
        session,
        request,
        event_type="monitoring.retraining_requested",
        resource_type="retraining_request",
        actor=user,
        resource_id=retraining.id,
        details={"drift_snapshot_id": str(snapshot.id)},
    )
    await session.commit()
    await session.refresh(retraining)
    return retraining


__all__ = ["router"]
