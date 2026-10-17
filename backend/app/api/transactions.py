"""Label-safe transaction ingestion and retrieval routes."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from backend.app.api.dependencies import CurrentUser, DatabaseSession, audit_request
from backend.app.db.models import Transaction
from backend.app.schemas.transaction import TransactionCreate, TransactionRead, mask_account


router = APIRouter(prefix="/transactions", tags=["transactions"])


@router.get("", response_model=list[TransactionRead])
async def list_transactions(
    session: DatabaseSession,
    user: CurrentUser,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0, le=1_000_000)] = 0,
    event_after: datetime | None = None,
    event_before: datetime | None = None,
) -> list[Transaction]:
    statement = select(Transaction)
    if event_after is not None:
        statement = statement.where(Transaction.event_time >= event_after)
    if event_before is not None:
        statement = statement.where(Transaction.event_time <= event_before)
    result = await session.execute(
        statement.order_by(Transaction.event_time.desc(), Transaction.id.desc())
        .offset(offset)
        .limit(limit)
    )
    return list(result.scalars())


@router.post("", response_model=TransactionRead, status_code=status.HTTP_201_CREATED)
async def create_transaction(
    payload: TransactionCreate,
    request: Request,
    session: DatabaseSession,
    user: CurrentUser,
) -> Transaction:
    transaction = Transaction(
        event_id=payload.event_id.strip(),
        dataset=payload.dataset.strip(),
        event_time=payload.event_time,
        masked_source_account=mask_account(payload.source_account_token),
        masked_destination_account=mask_account(payload.destination_account_token),
        transaction_type=payload.transaction_type.strip(),
        currency=payload.currency.strip().upper(),
        amount=payload.amount,
    )
    session.add(transaction)
    try:
        await session.flush()
    except IntegrityError:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="a transaction with this event_id already exists",
        ) from None
    await audit_request(
        session,
        request,
        event_type="transaction.created",
        resource_type="transaction",
        actor=user,
        resource_id=transaction.id,
        details={"dataset": transaction.dataset},
    )
    await session.commit()
    await session.refresh(transaction)
    return transaction


@router.get("/{transaction_id}", response_model=TransactionRead)
async def get_transaction(
    transaction_id: UUID,
    session: DatabaseSession,
    user: CurrentUser,
) -> Transaction:
    transaction = await session.get(Transaction, transaction_id)
    if transaction is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="transaction not found")
    return transaction


__all__ = ["router"]
