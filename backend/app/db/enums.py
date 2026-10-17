"""Shared database/domain enums with stable serialized values."""

from __future__ import annotations

from enum import StrEnum


class UserRole(StrEnum):
    FRAUD_ANALYST = "fraud_analyst"
    SENIOR_INVESTIGATOR = "senior_investigator"
    ADMIN = "admin"


class ModelStatus(StrEnum):
    CANDIDATE = "candidate"
    ACTIVE = "active"
    RETIRED = "retired"
    FAILED = "failed"


class CaseStatus(StrEnum):
    OPEN = "open"
    CLAIMED = "claimed"
    IN_REVIEW = "in_review"
    ESCALATED = "escalated"
    DISMISSED = "dismissed"
    MORE_INFO_REQUESTED = "more_info_requested"
    CLOSED = "closed"


class DispositionAction(StrEnum):
    ESCALATE = "escalate"
    DISMISS = "dismiss"
    REQUEST_MORE_INFO = "request_more_info"


class DriftStatus(StrEnum):
    STABLE = "stable"
    WATCH = "watch"
    DRIFT_DETECTED = "drift_detected"


class RetrainingStatus(StrEnum):
    OPEN = "open"
    APPROVED = "approved"
    RUNNING = "running"
    COMPLETED = "completed"
    REJECTED = "rejected"
    FAILED = "failed"


__all__ = [
    "CaseStatus",
    "DispositionAction",
    "DriftStatus",
    "ModelStatus",
    "RetrainingStatus",
    "UserRole",
]
