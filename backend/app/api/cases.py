"""Concurrent-safe analyst case workflow routes."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from backend.app.api.dependencies import DatabaseSession, audit_request, require_roles
from backend.app.db.enums import CaseStatus, DispositionAction, UserRole
from backend.app.db.models import (
    Case,
    CaseTransaction,
    Disposition,
    Prediction,
    Transaction,
    User,
)
from backend.app.schemas.case import (
    CaseClaim,
    CaseDetail,
    CaseDispositionCreate,
    CaseDispositionRead,
    CasePredictionRead,
    CaseQueueMaterializeRequest,
    CaseRead,
    CaseStartReview,
)
from backend.app.schemas.transaction import TransactionRead
from backend.app.services.case_management import InvalidCaseTransition, validate_transition
from backend.app.services.queue_materialization import (
    ActiveModelUnavailableError,
    materialize_hourly_queue,
)


router = APIRouter(prefix="/cases", tags=["cases"])
CaseUser = Annotated[
    User,
    Depends(
        require_roles(
            UserRole.FRAUD_ANALYST,
            UserRole.SENIOR_INVESTIGATOR,
            UserRole.ADMIN,
        )
    ),
]
QueueManager = Annotated[
    User,
    Depends(require_roles(UserRole.SENIOR_INVESTIGATOR, UserRole.ADMIN)),
]


@router.get("", response_model=list[CaseRead])
async def list_cases(
    session: DatabaseSession,
    user: CaseUser,
    case_status: CaseStatus | None = Query(default=None, alias="status"),
    assigned_to_me: bool = False,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0, le=1_000_000)] = 0,
) -> list[Case]:
    statement = select(Case)
    if case_status is not None:
        statement = statement.where(Case.status == case_status)
    if assigned_to_me:
        statement = statement.where(Case.assigned_to_id == user.id)
    result = await session.execute(
        statement.order_by(Case.alert_hour.asc(), Case.queue_rank.asc(), Case.id.asc())
        .offset(offset)
        .limit(limit)
    )
    return list(result.scalars())


@router.post("/materialize", response_model=list[CaseRead])
async def materialize_queue(
    payload: CaseQueueMaterializeRequest,
    request: Request,
    session: DatabaseSession,
    user: QueueManager,
) -> list[Case]:
    try:
        cases = await materialize_hourly_queue(
            session,
            alert_hour=payload.alert_hour,
            capacity=payload.capacity,
        )
    except ActiveModelUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    await audit_request(
        session,
        request,
        event_type="case.queue_materialized",
        resource_type="case_queue",
        actor=user,
        resource_id=payload.alert_hour.isoformat(),
        details={"capacity": payload.capacity, "selected_count": len(cases)},
    )
    await session.commit()
    return cases


@router.get("/{case_id}", response_model=CaseDetail)
async def get_case(
    case_id: UUID,
    session: DatabaseSession,
    user: CaseUser,
) -> CaseDetail:
    case = await _load_case_detail(session, case_id)
    if case is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="case not found")
    return _detail_response(case)


@router.post("/{case_id}/claim", response_model=CaseRead)
async def claim_case(
    case_id: UUID,
    payload: CaseClaim,
    request: Request,
    session: DatabaseSession,
    user: CaseUser,
) -> Case:
    case = await _locked_case(session, case_id)
    if case.status != payload.expected_status or case.status != CaseStatus.OPEN:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"case cannot be claimed from status {case.status.value}",
        )
    _transition(case, CaseStatus.CLAIMED)
    case.assigned_to_id = user.id
    case.claimed_at = datetime.now(timezone.utc)
    await audit_request(
        session,
        request,
        event_type="case.claimed",
        resource_type="case",
        actor=user,
        resource_id=case.id,
    )
    await session.commit()
    await session.refresh(case)
    return case


@router.post("/{case_id}/start-review", response_model=CaseRead)
async def start_review(
    case_id: UUID,
    payload: CaseStartReview,
    request: Request,
    session: DatabaseSession,
    user: CaseUser,
) -> Case:
    case = await _locked_case(session, case_id)
    _require_owner(case, user)
    if case.status != payload.expected_status:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"case status is {case.status.value}, not {payload.expected_status.value}",
        )
    _transition(case, CaseStatus.IN_REVIEW)
    await audit_request(
        session,
        request,
        event_type="case.review_started",
        resource_type="case",
        actor=user,
        resource_id=case.id,
    )
    await session.commit()
    await session.refresh(case)
    return case


@router.post(
    "/{case_id}/disposition",
    response_model=CaseDispositionRead,
    status_code=status.HTTP_201_CREATED,
)
async def disposition_case(
    case_id: UUID,
    payload: CaseDispositionCreate,
    request: Request,
    session: DatabaseSession,
    user: CaseUser,
) -> Disposition:
    case = await _locked_case(session, case_id)
    _require_owner(case, user)
    if case.status != CaseStatus.IN_REVIEW:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"case cannot be dispositioned from status {case.status.value}",
        )
    target = {
        DispositionAction.ESCALATE: CaseStatus.ESCALATED,
        DispositionAction.DISMISS: CaseStatus.DISMISSED,
        DispositionAction.REQUEST_MORE_INFO: CaseStatus.MORE_INFO_REQUESTED,
    }[payload.decision]
    _transition(case, target)
    if target in {CaseStatus.ESCALATED, CaseStatus.DISMISSED}:
        case.closed_at = datetime.now(timezone.utc)
    disposition = Disposition(
        case_id=case.id,
        analyst_id=user.id,
        action=payload.decision,
        confidence=payload.confidence / 5.0,
        notes=payload.rationale.strip() if payload.rationale else None,
        model_agreement=payload.model_agreement,
        is_override=payload.model_agreement is False,
        decision_seconds=payload.review_duration_seconds,
    )
    session.add(disposition)
    await session.flush()
    await audit_request(
        session,
        request,
        event_type="case.disposition_recorded",
        resource_type="case",
        actor=user,
        resource_id=case.id,
        details={"action": payload.decision.value},
    )
    await session.commit()
    await session.refresh(disposition)
    return disposition


async def _locked_case(session: DatabaseSession, case_id: UUID) -> Case:
    result = await session.execute(select(Case).where(Case.id == case_id).with_for_update())
    case = result.scalar_one_or_none()
    if case is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="case not found")
    return case


async def _load_case_detail(session: DatabaseSession, case_id: UUID) -> Case | None:
    result = await session.execute(
        select(Case)
        .where(Case.id == case_id)
        .options(
            selectinload(Case.transaction_links).selectinload(
                CaseTransaction.transaction
            ),
            selectinload(Case.transaction_links)
            .selectinload(CaseTransaction.prediction)
            .selectinload(Prediction.explanation),
            selectinload(Case.dispositions),
        )
    )
    return result.scalar_one_or_none()


def _detail_response(case: Case) -> CaseDetail:
    links = sorted(
        case.transaction_links,
        key=lambda item: (item.position, str(item.transaction_id)),
    )
    dispositions = sorted(
        case.dispositions,
        key=lambda item: (item.created_at, str(item.id)),
    )
    predictions = [
        CasePredictionRead.model_validate(link.prediction)
        for link in links
        if link.prediction is not None
    ]
    base = CaseRead.model_validate(case).model_dump()
    return CaseDetail(
        **base,
        transactions=[TransactionRead.model_validate(link.transaction) for link in links],
        predictions=predictions,
        dispositions=[CaseDispositionRead.model_validate(item) for item in dispositions],
    )


def _require_owner(case: Case, user: User) -> None:
    if case.assigned_to_id != user.id and user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="case is assigned to another analyst",
        )


def _transition(case: Case, target: CaseStatus) -> None:
    try:
        validate_transition(case.status.value, target.value)
    except (InvalidCaseTransition, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    case.status = target


__all__ = ["router"]
