"""Prediction request/response schemas for materialized causal features."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.app.services.inference import prohibited_input_fields


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transaction_id: UUID | None = None
    event_id: str | None = Field(default=None, max_length=128)
    features: dict[str, float]

    @model_validator(mode="after")
    def require_transaction_reference(self) -> "PredictionRequest":
        if self.transaction_id is None and not self.event_id:
            raise ValueError("transaction_id or event_id is required")
        return self

    @field_validator("features")
    @classmethod
    def validate_features(cls, value: dict[str, float]) -> dict[str, float]:
        if not value:
            raise ValueError("features cannot be empty")
        forbidden = prohibited_input_fields(value)
        if forbidden:
            raise ValueError(f"evaluation-only fields are prohibited: {sorted(forbidden)}")
        return value


class PredictionBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[PredictionRequest] = Field(min_length=1, max_length=256)


class ExplanationResponse(BaseModel):
    top_features: list[dict[str, Any]]
    narrative: str
    reliable: bool
    reliability_reasons: list[str]


class PredictionResponse(BaseModel):
    prediction_id: UUID | None = None
    transaction_id: UUID | None = None
    event_id: str | None = None
    supervised_probability: float = Field(ge=0, le=1)
    anomaly_probability: float | None = Field(default=None, ge=0, le=1)
    combined_risk: float = Field(ge=0, le=1)
    raw_margin: float
    model_version: str
    feature_schema_version: int
    explanation: ExplanationResponse
    scored_at: datetime


__all__ = [
    "ExplanationResponse",
    "PredictionBatchRequest",
    "PredictionRequest",
    "PredictionResponse",
]
