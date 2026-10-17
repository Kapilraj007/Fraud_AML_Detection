"""Persist an idempotent, capacity-limited hourly analyst queue."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.enums import CaseStatus, ModelStatus
from backend.app.db.models import (
    Case,
    CaseTransaction,
    ModelVersion,
    Prediction,
    Transaction,
)


STUDY_ARMS = (
    "data_only",
    "score_only",
    "generic_explanation",
    "reliability_gated_case_context",
)


class ActiveModelUnavailableError(RuntimeError):
    pass


async def materialize_hourly_queue(
    session: AsyncSession,
    *,
    alert_hour: datetime,
    capacity: int = 25,
) -> list[Case]:
    """Rank active-model scores for one UTC hour and persist top-N cases.

    Review identifiers and study arms are deterministic, making retries
    idempotent.  No labels, case truth, or evaluation metadata are read.
    """
    if capacity < 1 or capacity > 500:
        raise ValueError("capacity must be between 1 and 500")
    if alert_hour.tzinfo is None or alert_hour.utcoffset() is None:
        raise ValueError("alert_hour must include a timezone")
    start = alert_hour.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    end = start + timedelta(hours=1)
    # Serialize materialization for a model/hour on PostgreSQL.  The second
    # caller observes the deterministic rows created by the first instead of
    # racing the unique review_id constraint.  SQLite ignores FOR UPDATE,
    # which is sufficient for the single-process test backend.
    active_result = await session.execute(
        select(ModelVersion)
        .where(ModelVersion.status == ModelStatus.ACTIVE)
        .limit(1)
        .with_for_update()
    )
    active = active_result.scalar_one_or_none()
    if active is None:
        raise ActiveModelUnavailableError("no active model is registered")

    frozen_result = await session.execute(
        select(Case)
        .where(Case.alert_hour == start)
        .order_by(Case.queue_rank.asc(), Case.review_id.asc())
    )
    frozen = list(frozen_result.scalars())
    if frozen:
        return frozen

    result = await session.execute(
        select(Prediction, Transaction)
        .join(Transaction, Transaction.id == Prediction.transaction_id)
        .where(
            Prediction.model_version_id == active.id,
            Transaction.event_time >= start,
            Transaction.event_time < end,
        )
        .order_by(Prediction.combined_score.desc(), Transaction.event_id.asc())
        .limit(capacity)
    )
    ranked = list(result.all())
    review_ids = [
        _review_id(active.id, start, transaction.event_id)
        for _, transaction in ranked
    ]
    existing: dict[str, Case] = {}
    if review_ids:
        rows = await session.execute(select(Case).where(Case.review_id.in_(review_ids)))
        existing = {case.review_id: case for case in rows.scalars()}

    cases: list[Case] = []
    for rank, ((prediction, transaction), review_id) in enumerate(
        zip(ranked, review_ids, strict=True), start=1
    ):
        case = existing.get(review_id)
        if case is None:
            case = Case(
                review_id=review_id,
                status=CaseStatus.OPEN,
                queue_rank=rank,
                alert_hour=start,
                risk_score=prediction.combined_score,
                study_arm=_study_arm(review_id),
            )
            session.add(case)
            await session.flush()
            session.add(
                CaseTransaction(
                    case_id=case.id,
                    transaction_id=transaction.id,
                    prediction_id=prediction.id,
                    position=0,
                )
            )
        cases.append(case)
    await session.flush()
    return cases


def _review_id(model_version_id: object, hour: datetime, event_id: str) -> str:
    value = f"{model_version_id}|{hour.isoformat()}|{event_id}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]


def _study_arm(review_id: str) -> str:
    index = int(hashlib.sha256(f"arm|{review_id}".encode("utf-8")).hexdigest(), 16)
    return STUDY_ARMS[index % len(STUDY_ARMS)]


__all__ = [
    "ActiveModelUnavailableError",
    "STUDY_ARMS",
    "materialize_hourly_queue",
]
