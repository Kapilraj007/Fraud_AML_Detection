"""Case workflow invariants shared by API and persistence layers."""

from __future__ import annotations

from enum import StrEnum


class CaseStatus(StrEnum):
    OPEN = "open"
    CLAIMED = "claimed"
    IN_REVIEW = "in_review"
    MORE_INFO_REQUESTED = "more_info_requested"
    ESCALATED = "escalated"
    DISMISSED = "dismissed"


class DispositionDecision(StrEnum):
    ESCALATE = "escalate"
    DISMISS = "dismiss"
    REQUEST_MORE_INFO = "request_more_info"


TRANSITIONS: dict[CaseStatus, frozenset[CaseStatus]] = {
    CaseStatus.OPEN: frozenset({CaseStatus.CLAIMED}),
    CaseStatus.CLAIMED: frozenset({CaseStatus.IN_REVIEW}),
    CaseStatus.IN_REVIEW: frozenset(
        {
            CaseStatus.MORE_INFO_REQUESTED,
            CaseStatus.ESCALATED,
            CaseStatus.DISMISSED,
        }
    ),
    CaseStatus.MORE_INFO_REQUESTED: frozenset({CaseStatus.IN_REVIEW}),
    CaseStatus.ESCALATED: frozenset(),
    CaseStatus.DISMISSED: frozenset(),
}


class InvalidCaseTransition(ValueError):
    pass


def validate_transition(current: str | CaseStatus, requested: str | CaseStatus) -> CaseStatus:
    source = CaseStatus(current)
    target = CaseStatus(requested)
    if target not in TRANSITIONS[source]:
        raise InvalidCaseTransition(f"case cannot transition from {source} to {target}")
    return target


def disposition_status(decision: str | DispositionDecision) -> CaseStatus:
    value = DispositionDecision(decision)
    return {
        DispositionDecision.ESCALATE: CaseStatus.ESCALATED,
        DispositionDecision.DISMISS: CaseStatus.DISMISSED,
        DispositionDecision.REQUEST_MORE_INFO: CaseStatus.MORE_INFO_REQUESTED,
    }[value]


__all__ = [
    "CaseStatus",
    "DispositionDecision",
    "InvalidCaseTransition",
    "disposition_status",
    "validate_transition",
]
