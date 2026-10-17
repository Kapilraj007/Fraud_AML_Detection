"""Capacity-constrained, label-safe alert queue construction.

Sprint 1 has no trained model, so the harness accepts a caller-supplied score
or uses a fixed, label-blind development score. Ground truth is split into a
separate evaluation key before the analyst-facing queue is returned.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_ARMS = (
    "data_only",
    "score_only",
    "generic_explanation",
    "reliability_gated_case_context",
)
FORBIDDEN_SCORE_NAMES = {
    "label", "target", "groundtruth", "resolvedlabel", "isfraud",
    "islaundering", "fraudlabel", "launderinglabel",
}
SEALED_METADATA_NAMES = {
    "labelname", "caseid", "caseids", "typology", "typologies",
    "patternmatched", "patternid", "patterndetail", "patternlinenumber",
    "caseposition", "casedetail", "launderingcaseids",
    "launderingtypologies", "launderingcasepositions",
    "launderingcasecount",
}


@dataclass(frozen=True)
class QueueConfig:
    capacity_per_hour: int = 25
    score_column: str | None = None
    study_seed: int = 20261007
    study_arms: tuple[str, ...] = DEFAULT_ARMS
    minutes_per_review: float = 5.0

    def __post_init__(self) -> None:
        if self.capacity_per_hour < 1:
            raise ValueError("capacity_per_hour must be positive")
        if not self.study_arms or len(set(self.study_arms)) != len(self.study_arms):
            raise ValueError("study_arms must be non-empty and unique")
        if self.minutes_per_review <= 0:
            raise ValueError("minutes_per_review must be positive")


@dataclass(frozen=True)
class QueueBundle:
    alerts: pd.DataFrame
    evaluation_key: pd.DataFrame


def build_case_queue(
    frame: pd.DataFrame, config: QueueConfig | None = None
) -> QueueBundle:
    """Rank candidates per hour and isolate ground truth from reviewers."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    cfg = config or QueueConfig()
    if frame.empty:
        return QueueBundle(pd.DataFrame(), pd.DataFrame())
    if "event_id" not in frame:
        raise ValueError("frame must contain event_id")
    if frame["event_id"].astype(str).duplicated().any():
        raise ValueError("event_id must be unique")

    work = frame.copy()
    label_column = _label_column(work)
    case_ids = _case_memberships(work)
    typologies = _typology_memberships(work)
    score, source = _score(work, cfg.score_column, label_column)
    work["risk_score"] = score
    work["score_source"] = source
    work["alert_hour"], work["_queue_time_hours"] = _hour_bucket(work)
    work["_event_tie"] = work["event_id"].astype(str)
    work = work.sort_values(
        ["alert_hour", "risk_score", "_event_tie"],
        ascending=[True, False, True], kind="mergesort",
    )
    # Keep sealed annotations aligned with the stable queue order.  ``sort_values``
    # preserves original indices, which makes this reindex deterministic.
    case_ids = case_ids.reindex(work.index)
    typologies = typologies.reindex(work.index)
    work["queue_rank"] = work.groupby("alert_hour", sort=False).cumcount() + 1
    work["selected_by_capacity"] = work["queue_rank"] <= cfg.capacity_per_hour
    work["review_id"] = work["event_id"].map(
        lambda value: _digest(f"review|{cfg.study_seed}|{value}")[:20]
    )
    work["study_arm"] = _assign_arms(work, cfg, case_ids)
    if "deferral_eligible" not in work:
        if "uncertainty" in work:
            uncertainty = pd.to_numeric(work["uncertainty"], errors="coerce")
            work["deferral_eligible"] = uncertainty.ge(0.5).fillna(False)
        else:
            work["deferral_eligible"] = False
    if "drift_regime" not in work:
        work["drift_regime"] = "unassigned"

    key_columns = [
        "review_id", "event_id", "alert_hour", "_queue_time_hours",
        "selected_by_capacity", "queue_rank", "risk_score", "study_arm",
    ]
    evaluation = work.loc[:, key_columns].copy()
    evaluation["label"] = (
        pd.to_numeric(work[label_column], errors="coerce").fillna(0).astype(int)
        if label_column else 0
    )
    evaluation["case_ids"] = [
        json.dumps(list(values), separators=(",", ":")) for values in case_ids
    ]
    evaluation["case_id"] = [values[0] if values else pd.NA for values in case_ids]
    evaluation["typologies"] = [
        json.dumps(list(values), separators=(",", ":")) for values in typologies
    ]
    evaluation["typology"] = [
        values[0] if values else pd.NA for values in typologies
    ]
    if "amount" in work:
        evaluation["amount"] = work["amount"].to_numpy()

    unsafe = {
        column for column in work
        if _normalise(column) in FORBIDDEN_SCORE_NAMES | SEALED_METADATA_NAMES
    }
    if label_column:
        unsafe.add(label_column)
    unsafe.update({"_queue_time_hours", "_event_tie"})
    public = work.loc[work["selected_by_capacity"]].drop(
        columns=[column for column in unsafe if column in work], errors="ignore"
    )
    public = public.reset_index(drop=True)
    evaluation = evaluation.reset_index(drop=True)
    return QueueBundle(public, evaluation)


def evaluate_queue(
    bundle: QueueBundle, *, minutes_per_review: float = 5.0
) -> dict[str, float | int]:
    """Compute queue-level outcomes using the sealed evaluation key."""
    if minutes_per_review <= 0:
        raise ValueError("minutes_per_review must be positive")
    key = bundle.evaluation_key.copy()
    if key.empty:
        return {
            "alerts_reviewed": 0, "alert_precision": 0.0,
            "alert_recall": 0.0, "case_recall": 0.0,
            "false_escalations": 0, "reviewed_per_hour": 0.0,
            "time_to_first_detection_hours": float("nan"),
            "positive_backlog_age_hours": 0.0,
            "laundering_value_per_analyst_hour": 0.0,
        }
    selected = key[key["selected_by_capacity"]]
    positive = key[key["label"].eq(1)]
    found = selected[selected["label"].eq(1)]
    reviewed = len(selected)
    precision = len(found) / reviewed if reviewed else 0.0
    recall = len(found) / len(positive) if len(positive) else 0.0
    all_cases = _membership_union(positive.get("case_ids"))
    found_cases = _membership_union(found.get("case_ids"))
    case_recall = len(found_cases) / len(all_cases) if all_cases else recall
    hourly = selected.groupby("alert_hour").size()
    start = float(key["_queue_time_hours"].min())
    first = (
        float(found["_queue_time_hours"].min()) - start if len(found)
        else float("nan")
    )
    missed = positive[~positive["selected_by_capacity"]]
    end = float(key["_queue_time_hours"].max())
    backlog_age = float((end - missed["_queue_time_hours"]).mean()) if len(missed) else 0.0
    captured_value = 0.0
    if "amount" in found:
        captured_value = float(pd.to_numeric(found["amount"], errors="coerce").fillna(0).sum())
    analyst_hours = reviewed * minutes_per_review / 60.0
    return {
        "alerts_reviewed": reviewed,
        "alert_precision": precision,
        "alert_recall": recall,
        "case_recall": case_recall,
        "false_escalations": int(reviewed - len(found)),
        "reviewed_per_hour": float(hourly.mean()) if len(hourly) else 0.0,
        "time_to_first_detection_hours": first,
        "positive_backlog_age_hours": backlog_age,
        "laundering_value_per_analyst_hour": (
            captured_value / analyst_hours if analyst_hours else 0.0
        ),
    }


def _score(
    frame: pd.DataFrame, score_column: str | None, label_column: str | None
) -> tuple[pd.Series, str]:
    if score_column is not None:
        if score_column not in frame:
            raise ValueError(f"score column not found: {score_column}")
        if _normalise(score_column) in FORBIDDEN_SCORE_NAMES or score_column == label_column:
            raise ValueError("ground truth cannot be used as a queue score")
        values = pd.to_numeric(frame[score_column], errors="coerce")
        return values.replace([np.inf, -np.inf], np.nan).fillna(-1.0), score_column

    def numeric(name: str) -> pd.Series:
        if name not in frame:
            return pd.Series(0.0, index=frame.index)
        return pd.to_numeric(frame[name], errors="coerce").fillna(0.0)

    amount = np.clip(np.log1p(numeric("amount").clip(lower=0)) / 20.0, 0, 1)
    velocity = np.clip(np.log1p(numeric("source_txn_count_1h")) / np.log(21), 0, 1)
    inbound = np.clip(np.log1p(numeric("source_recent_inbound_count")) / np.log(11), 0, 1)
    pass_through = np.clip(numeric("rapid_pass_through_ratio") / 2.0, 0, 1)
    new_party = numeric("is_new_counterparty").clip(0, 1)
    score = 0.25 * amount + 0.25 * velocity + 0.2 * inbound + 0.2 * pass_through + 0.1 * new_party
    return pd.Series(score, index=frame.index, dtype=float), "label_blind_development_rule_v1"


def _hour_bucket(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    if "event_time" in frame and pd.to_datetime(frame["event_time"], errors="coerce", utc=True).notna().any():
        timestamp = pd.to_datetime(frame["event_time"], errors="coerce", utc=True)
        if timestamp.isna().any():
            raise ValueError("event_time cannot mix known and missing values")
        hour = timestamp.dt.floor("h")
        numeric = hour.astype("int64") / 3_600_000_000_000
        return hour.astype(str), pd.Series(numeric, index=frame.index, dtype=float)
    if "relative_time_seconds" not in frame:
        raise ValueError("event_time or relative_time_seconds is required")
    relative = pd.to_numeric(frame["relative_time_seconds"], errors="coerce")
    if relative.isna().any():
        raise ValueError("relative_time_seconds contains missing values")
    hour_number = np.floor(relative / 3600).astype(int)
    return hour_number.map(lambda value: f"relative-hour-{value:08d}"), relative / 3600.0


def _assign_arms(
    frame: pd.DataFrame,
    config: QueueConfig,
    case_ids: pd.Series | None = None,
) -> pd.Series:
    memberships = _case_memberships(frame) if case_ids is None else case_ids
    unit = pd.Series(
        (
            values[0] if values else str(event_id)
            for values, event_id in zip(memberships, frame["event_id"])
        ),
        index=frame.index,
        dtype="object",
    )
    units = sorted(set(unit), key=lambda value: _digest(f"arm|{config.study_seed}|{value}"))
    mapping = {value: config.study_arms[index % len(config.study_arms)] for index, value in enumerate(units)}
    return unit.map(mapping)


def _case_memberships(frame: pd.DataFrame) -> pd.Series:
    if "laundering_case_ids" in frame:
        source = frame["laundering_case_ids"]
    elif "case_ids" in frame:
        source = frame["case_ids"]
    elif "case_id" in frame:
        source = frame["case_id"]
    else:
        source = pd.Series([pd.NA] * len(frame), index=frame.index, dtype="object")
    return pd.Series((_memberships(value) for value in source), index=frame.index)


def _typology_memberships(frame: pd.DataFrame) -> pd.Series:
    if "laundering_typologies" in frame:
        source = frame["laundering_typologies"]
    elif "typologies" in frame:
        source = frame["typologies"]
    elif "typology" in frame:
        source = frame["typology"]
    else:
        source = pd.Series([pd.NA] * len(frame), index=frame.index, dtype="object")
    return pd.Series((_memberships(value) for value in source), index=frame.index)


def _memberships(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    try:
        if bool(pd.isna(value)):
            return ()
    except (TypeError, ValueError):
        pass
    parsed: object = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ()
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = value
    if isinstance(parsed, (list, tuple, set)):
        values = [str(item).strip() for item in parsed if str(item).strip()]
    else:
        values = [str(parsed).strip()]
    return tuple(dict.fromkeys(values))


def _membership_union(values: pd.Series | None) -> set[str]:
    if values is None:
        return set()
    answer: set[str] = set()
    for value in values:
        answer.update(_memberships(value))
    return answer


def _label_column(frame: pd.DataFrame) -> str | None:
    declared = frame.attrs.get("label_column")
    if isinstance(declared, str) and declared in frame:
        return declared
    for column in frame:
        if _normalise(column) in FORBIDDEN_SCORE_NAMES:
            return column
    return None


def _normalise(value: object) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = [
    "QueueBundle", "QueueConfig", "build_case_queue", "evaluate_queue",
]
