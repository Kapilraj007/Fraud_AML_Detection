"""Model registry API schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelVersionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    version: str
    name: str
    algorithm: str
    artifact_uri: str
    artifact_sha256: str
    feature_schema: dict[str, Any]
    metrics: dict[str, Any] = Field(default_factory=dict)
    status: str
    created_at: datetime
    activated_at: datetime | None = None


class ModelPromotionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_current_version: str | None = None
    reason: str = Field(min_length=8, max_length=2000)


__all__ = ["ModelPromotionRequest", "ModelVersionRead"]
