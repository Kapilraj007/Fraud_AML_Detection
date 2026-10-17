"""Drift status and human-gated retraining request schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from backend.app.db.enums import DriftStatus, RetrainingStatus


class DriftSnapshotRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    model_version_id: UUID
    window_start: datetime
    window_end: datetime
    status: DriftStatus
    adwin_detected: bool
    feature_metrics: dict[str, Any] = Field(default_factory=dict)
    prediction_metrics: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class RetrainingRequestCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = Field(min_length=8, max_length=2000)
    drift_snapshot_id: UUID


class RetrainingRequestRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    status: RetrainingStatus
    reason: str
    requested_by_id: UUID
    drift_snapshot_id: UUID
    created_at: datetime
    updated_at: datetime


class MonitoringStatusRead(BaseModel):
    active_model_id: UUID
    active_model_version: str
    status: DriftStatus | None
    latest_snapshot: DriftSnapshotRead | None = None


__all__ = [
    "DriftSnapshotRead",
    "MonitoringStatusRead",
    "RetrainingRequestCreate",
    "RetrainingRequestRead",
]
