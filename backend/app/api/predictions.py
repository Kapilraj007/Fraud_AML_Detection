"""Online scoring routes backed by the verified active model bundle."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from backend.app.api.dependencies import CurrentUser, DatabaseSession, Inference, audit_request
from backend.app.db.enums import DriftStatus
from backend.app.db.models import Explanation, Prediction, Transaction
from backend.app.db.repositories import get_active_model_version
from backend.app.schemas.prediction import (
    ExplanationResponse,
    PredictionBatchRequest,
    PredictionRequest,
    PredictionResponse,
)
from backend.app.services.inference import ScoreResult


router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.post("/score", response_model=PredictionResponse, status_code=status.HTTP_201_CREATED)
async def score_transaction(
    payload: PredictionRequest,
    request: Request,
    session: DatabaseSession,
    user: CurrentUser,
    inference: Inference,
) -> PredictionResponse:
    active_model = await get_active_model_version(session)
    if active_model is None:
        raise _model_unavailable("no active model is registered")
    transaction = await _resolve_transaction(session, payload)
    await _ensure_not_scored(session, transaction.id, active_model.id)
    result = _score_one(inference, payload.features)
    prediction = _prediction_record(transaction.id, active_model, result, payload.features)
    session.add(prediction)
    try:
        await session.flush()
    except IntegrityError:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="this transaction was already scored by the active model",
        ) from None
    await audit_request(
        session,
        request,
        event_type="prediction.scored",
        resource_type="prediction",
        actor=user,
        resource_id=prediction.id,
        details={"model_version_id": str(active_model.id), "batch_size": 1},
    )
    await session.commit()
    return _response(prediction, transaction, result)


@router.post(
    "/batch-score",
    response_model=list[PredictionResponse],
    status_code=status.HTTP_201_CREATED,
)
@router.post(
    "/score/batch",
    response_model=list[PredictionResponse],
    status_code=status.HTTP_201_CREATED,
)
async def score_batch(
    payload: PredictionBatchRequest,
    request: Request,
    session: DatabaseSession,
    user: CurrentUser,
    inference: Inference,
) -> list[PredictionResponse]:
    active_model = await get_active_model_version(session)
    if active_model is None:
        raise _model_unavailable("no active model is registered")
    transactions = [await _resolve_transaction(session, row) for row in payload.rows]
    transaction_ids = [item.id for item in transactions]
    if len(transaction_ids) != len(set(transaction_ids)):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="a batch cannot score the same transaction more than once",
        )
    existing = await session.execute(
        select(Prediction.transaction_id).where(
            Prediction.model_version_id == active_model.id,
            Prediction.transaction_id.in_(transaction_ids),
        )
    )
    already_scored = set(existing.scalars())
    if already_scored:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="one or more transactions were already scored by the active model",
        )
    try:
        results = inference.score_batch([row.features for row in payload.rows])
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    records: list[Prediction] = []
    for transaction, row, result in zip(transactions, payload.rows, results, strict=True):
        record = _prediction_record(transaction.id, active_model, result, row.features)
        session.add(record)
        records.append(record)
    try:
        await session.flush()
    except IntegrityError:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="one or more transactions were already scored by the active model",
        ) from None
    await audit_request(
        session,
        request,
        event_type="prediction.batch_scored",
        resource_type="prediction_batch",
        actor=user,
        details={"model_version_id": str(active_model.id), "batch_size": len(records)},
    )
    await session.commit()
    return [
        _response(record, transaction, result)
        for record, transaction, result in zip(records, transactions, results, strict=True)
    ]


async def _resolve_transaction(session: DatabaseSession, payload: PredictionRequest) -> Transaction:
    transaction: Transaction | None = None
    if payload.transaction_id is not None:
        transaction = await session.get(Transaction, payload.transaction_id)
    if payload.event_id:
        result = await session.execute(
            select(Transaction).where(Transaction.event_id == payload.event_id)
        )
        by_event = result.scalar_one_or_none()
        if transaction is not None and by_event is not None and transaction.id != by_event.id:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="transaction_id and event_id refer to different transactions",
            )
        transaction = transaction or by_event
    if transaction is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="transaction not found")
    return transaction


async def _ensure_not_scored(
    session: DatabaseSession, transaction_id: UUID, model_version_id: UUID
) -> None:
    result = await session.execute(
        select(Prediction.id).where(
            Prediction.transaction_id == transaction_id,
            Prediction.model_version_id == model_version_id,
        )
    )
    if result.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="this transaction was already scored by the active model",
        )


def _score_one(inference: object, features: dict[str, float]) -> ScoreResult:
    try:
        return inference.score_one(features)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


def _prediction_record(
    transaction_id: UUID,
    model: object,
    result: ScoreResult,
    features: dict[str, float],
) -> Prediction:
    reliable = bool(result.explanation.get("reliable", False))
    reasons = [str(item) for item in result.explanation.get("reliability_reasons", [])]
    uncertainty = 1.0 - min(1.0, abs(result.combined_risk - 0.5) * 2.0)
    schema_hash = hashlib.sha256(
        json.dumps(sorted(features), separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    prediction = Prediction(
        transaction_id=transaction_id,
        model_version_id=model.id,
        supervised_score=result.supervised_probability,
        anomaly_score=(
            result.supervised_probability
            if result.anomaly_probability is None
            else result.anomaly_probability
        ),
        combined_score=result.combined_risk,
        decision_threshold=model.decision_threshold,
        predicted_positive=result.combined_risk >= model.decision_threshold,
        uncertainty=uncertainty,
        reliability_score=1.0 if reliable else 0.0,
        reliability_status="reliable" if reliable else "withheld",
        explanation_gate_reason="; ".join(reasons) or None,
        drift_status=DriftStatus.STABLE,
        feature_schema_hash=schema_hash,
    )
    prediction.explanation = Explanation(
        top_features=list(result.explanation.get("top_features", [])),
        narrative=str(result.explanation.get("narrative", "")),
        is_withheld=not reliable,
        gate_reason="; ".join(reasons) or None,
    )
    return prediction


def _response(
    prediction: Prediction,
    transaction: Transaction,
    result: ScoreResult,
) -> PredictionResponse:
    return PredictionResponse(
        prediction_id=prediction.id,
        transaction_id=transaction.id,
        event_id=transaction.event_id,
        supervised_probability=result.supervised_probability,
        anomaly_probability=result.anomaly_probability,
        combined_risk=result.combined_risk,
        raw_margin=result.raw_margin,
        model_version=result.model_version,
        feature_schema_version=result.feature_schema_version,
        explanation=ExplanationResponse.model_validate(result.explanation),
        scored_at=prediction.created_at or datetime.now(timezone.utc),
    )


def _model_unavailable(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail)


__all__ = ["router"]
