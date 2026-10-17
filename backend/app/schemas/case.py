"""Analyst case queue and append-only disposition schemas."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.app.db.enums import CaseStatus, DispositionAction, DriftStatus
from backend.app.schemas.transaction import TransactionRead


class CaseRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    review_id: str
    status: CaseStatus
    queue_rank: int
    risk_score: float = Field(ge=0, le=1)
    assigned_to_id: UUID | None = None
    study_arm: str
    alert_hour: datetime
    updated_at: datetime
    created_at: datetime


class CaseClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_status: CaseStatus = CaseStatus.OPEN


class CaseStartReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_status: CaseStatus = CaseStatus.CLAIMED


class CaseQueueMaterializeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alert_hour: datetime
    capacity: int = Field(default=25, ge=1, le=500)

    @field_validator("alert_hour")
    @classmethod
    def normalize_alert_hour(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("alert_hour must include a timezone")
        return value.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)


class CaseDispositionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: DispositionAction
    confidence: int = Field(ge=1, le=5)
    rationale: str | None = Field(default=None, max_length=4000)
    model_agreement: bool | None = None
    review_duration_seconds: int = Field(ge=0, le=86400)


class CaseDispositionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    case_id: UUID
    analyst_id: UUID
    action: DispositionAction
    confidence: float | None
    notes: str | None
    model_agreement: bool | None
    decision_seconds: int | None
    created_at: datetime


class CaseExplanationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    top_features: list[dict[str, Any]] = Field(default_factory=list)
    narrative: str | None
    is_withheld: bool
    gate_reason: str | None


class CasePredictionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    transaction_id: UUID
    model_version_id: UUID
    supervised_score: float = Field(ge=0, le=1)
    anomaly_score: float = Field(ge=0, le=1)
    combined_score: float = Field(ge=0, le=1)
    decision_threshold: float = Field(ge=0, le=1)
    predicted_positive: bool
    uncertainty: float = Field(ge=0, le=1)
    reliability_score: float = Field(ge=0, le=1)
    reliability_status: str
    explanation_gate_reason: str | None
    drift_status: DriftStatus
    explanation: CaseExplanationRead | None = None
    created_at: datetime


class CaseDetail(CaseRead):
    transactions: list[TransactionRead] = Field(default_factory=list)
    predictions: list[CasePredictionRead] = Field(default_factory=list)
    dispositions: list[CaseDispositionRead] = Field(default_factory=list)


__all__ = [
    "CaseClaim",
    "CaseDetail",
    "CaseDispositionCreate",
    "CaseDispositionRead",
    "CaseExplanationRead",
    "CasePredictionRead",
    "CaseRead",
    "CaseQueueMaterializeRequest",
    "CaseStartReview",
]
