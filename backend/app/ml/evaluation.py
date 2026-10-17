"""Leakage-safe classification and capacity-constrained AML evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import json
import math
from typing import Any

import numpy as np
import pandas as pd

from backend.app.ml.model import ModelInputError, validate_binary_labels


def average_precision(labels: Sequence[Any], scores: Sequence[float]) -> float:
    """Compute non-interpolated average precision with stable score ordering."""
    target, risk = _validated_pair(labels, scores)
    order = np.argsort(-risk, kind="mergesort")
    ordered = target[order]
    cumulative = np.cumsum(ordered)
    positive_positions = np.flatnonzero(ordered == 1)
    return float(np.mean(cumulative[positive_positions] / (positive_positions + 1)))


def roc_auc(labels: Sequence[Any], scores: Sequence[float]) -> float:
    """Compute AUROC using average ranks for tied scores."""
    target, risk = _validated_pair(labels, scores)
    ranks = pd.Series(risk).rank(method="average").to_numpy(dtype=float)
    positives = target == 1
    positive_count = int(positives.sum())
    negative_count = len(target) - positive_count
    statistic = ranks[positives].sum() - positive_count * (positive_count + 1) / 2
    return float(statistic / (positive_count * negative_count))


def classification_metrics(
    labels: Sequence[Any],
    probabilities: Sequence[float],
    *,
    threshold: float,
    calibration_bins: int = 10,
) -> dict[str, Any]:
    """Return JSON-safe discrimination, threshold, and calibration metrics."""
    target, probability = _validated_pair(labels, probabilities, probability=True)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between zero and one")
    prediction = probability >= threshold
    positive = target == 1
    true_positive = int(np.sum(prediction & positive))
    false_positive = int(np.sum(prediction & ~positive))
    false_negative = int(np.sum(~prediction & positive))
    true_negative = int(np.sum(~prediction & ~positive))
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    brier = float(np.mean((probability - target) ** 2))
    calibration = calibration_summary(target, probability, bins=calibration_bins)
    return {
        "rows": len(target),
        "positives": int(positive.sum()),
        "positive_rate": float(positive.mean()),
        "average_precision": average_precision(target, probability),
        "pr_auc": average_precision(target, probability),
        "roc_auc": roc_auc(target, probability),
        "brier_score": brier,
        "expected_calibration_error": calibration["expected_calibration_error"],
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positive,
        "false_positives": false_positive,
        "false_negatives": false_negative,
        "true_negatives": true_negative,
        "calibration_bins": calibration["bins"],
    }


def calibration_summary(
    labels: Sequence[Any], probabilities: Sequence[float], *, bins: int = 10
) -> dict[str, Any]:
    target, probability = _validated_pair(labels, probabilities, probability=True)
    if bins < 2:
        raise ValueError("at least two calibration bins are required")
    # Equal-width bins are deterministic and preserve comparability across runs.
    assignments = np.minimum((probability * bins).astype(int), bins - 1)
    records: list[dict[str, Any]] = []
    error = 0.0
    for index in range(bins):
        selected = assignments == index
        count = int(selected.sum())
        if count:
            mean_probability = float(probability[selected].mean())
            observed_rate = float(target[selected].mean())
            error += (count / len(target)) * abs(mean_probability - observed_rate)
        else:
            mean_probability = None
            observed_rate = None
        records.append(
            {
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "rows": count,
                "mean_probability": mean_probability,
                "observed_rate": observed_rate,
            }
        )
    return {"expected_calibration_error": float(error), "bins": records}


def choose_threshold(
    labels: Sequence[Any],
    probabilities: Sequence[float],
    *,
    beta: float = 1.0,
) -> dict[str, float]:
    """Choose the validation-only threshold maximizing F-beta."""
    target, probability = _validated_pair(labels, probabilities, probability=True)
    if beta <= 0:
        raise ValueError("beta must be positive")
    order = np.argsort(-probability, kind="mergesort")
    ordered_target = target[order]
    ordered_probability = probability[order]
    true_positive = np.cumsum(ordered_target)
    predicted_positive = np.arange(1, len(target) + 1)
    precision = true_positive / predicted_positive
    recall = true_positive / int(target.sum())
    beta_squared = beta * beta
    denominator = beta_squared * precision + recall
    f_beta = np.divide(
        (1 + beta_squared) * precision * recall,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator > 0,
    )
    # Only evaluate a threshold after the final row in a tied-score group.
    ends = np.r_[ordered_probability[1:] != ordered_probability[:-1], True]
    candidates = np.flatnonzero(ends)
    best = candidates[int(np.argmax(f_beta[candidates]))]
    threshold = float(ordered_probability[best])
    return {
        "threshold": threshold,
        "f_beta": float(f_beta[best]),
        "beta": float(beta),
        "precision": float(precision[best]),
        "recall": float(recall[best]),
    }


def capacity_metrics(
    labels: Sequence[Any],
    scores: Sequence[float],
    event_times: Sequence[Any],
    event_ids: Sequence[Any],
    *,
    capacity_per_hour: int,
    case_ids: Sequence[Any] | None = None,
    known_case_ids: Iterable[str] | None = None,
    amounts: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Evaluate deterministic top-N-per-hour review capacity.

    Labels and optional case identifiers are consumed only after ranking.  The
    ranking contract is score descending, then event ID ascending.
    """
    target, risk = _validated_pair(labels, scores)
    if capacity_per_hour < 1:
        raise ValueError("capacity_per_hour must be positive")
    if len(event_times) != len(target) or len(event_ids) != len(target):
        raise ValueError("event metadata row counts differ from predictions")
    times = pd.to_datetime(pd.Series(event_times), errors="coerce", utc=True)
    if times.isna().any():
        raise ValueError("event_times contains invalid timestamps")
    identifiers = pd.Series(event_ids, dtype="string")
    if identifiers.isna().any() or identifiers.duplicated().any():
        raise ValueError("event_ids must be complete and unique")
    frame = pd.DataFrame(
        {
            "label": target,
            "score": risk,
            "event_time": times,
            "event_id": identifiers,
            "event_hour": times.dt.floor("h"),
            "position": np.arange(len(target)),
        }
    )
    ranked = frame.sort_values(
        ["event_hour", "score", "event_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    selected = ranked.groupby("event_hour", sort=True, group_keys=False).head(
        int(capacity_per_hour)
    )
    selected_positions = selected["position"].to_numpy(dtype=int)
    selected_labels = target[selected_positions]
    alerts = len(selected)
    reviewed_positives = int(selected_labels.sum())
    total_positives = int(target.sum())
    result: dict[str, Any] = {
        "capacity_per_hour": int(capacity_per_hour),
        "active_hours": int(frame["event_hour"].nunique()),
        "alerts_reviewed": alerts,
        "reviewed_per_active_hour": _safe_divide(
            alerts, int(frame["event_hour"].nunique())
        ),
        "true_positive_alerts": reviewed_positives,
        "false_escalations": alerts - reviewed_positives,
        "alert_precision": _safe_divide(reviewed_positives, alerts),
        "alert_recall": _safe_divide(reviewed_positives, total_positives),
    }

    if amounts is not None:
        numeric_amounts = pd.to_numeric(pd.Series(amounts), errors="coerce").to_numpy()
        if len(numeric_amounts) != len(target):
            raise ValueError("amount row count differs from predictions")
        recovered = float(np.nansum(numeric_amounts[selected_positions][selected_labels == 1]))
        result["recovered_laundering_value"] = recovered
        result["laundering_value_per_active_hour"] = _safe_divide(
            recovered, result["active_hours"]
        )

    if case_ids is not None:
        if len(case_ids) != len(target):
            raise ValueError("case ID row count differs from predictions")
        parsed = [_parse_case_ids(value) for value in case_ids]
        all_cases = {
            case
            for row, label in zip(parsed, target)
            if label == 1
            for case in row
        }
        selected_cases = {
            case
            for position in selected_positions
            if target[position] == 1
            for case in parsed[position]
        }
        result.update(
            {
                "test_cases": len(all_cases),
                "detected_cases": len(selected_cases & all_cases),
                "case_recall": _safe_divide(len(selected_cases & all_cases), len(all_cases)),
            }
        )
        if known_case_ids is not None:
            known = {str(value) for value in known_case_ids}
            first_seen = all_cases - known
            result.update(
                {
                    "first_seen_test_cases": len(first_seen),
                    "detected_first_seen_cases": len(selected_cases & first_seen),
                    "first_seen_case_recall": _safe_divide(
                        len(selected_cases & first_seen), len(first_seen)
                    ),
                }
            )
    return result


def evaluate_predictions(
    labels: Sequence[Any],
    probabilities: Sequence[float],
    *,
    threshold: float,
    event_times: Sequence[Any] | None = None,
    event_ids: Sequence[Any] | None = None,
    capacity_per_hour: int | None = None,
    case_ids: Sequence[Any] | None = None,
    known_case_ids: Iterable[str] | None = None,
    amounts: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a frozen prediction vector; test data remain read-only."""
    metrics = classification_metrics(labels, probabilities, threshold=threshold)
    supplied_capacity_fields = (event_times, event_ids, capacity_per_hour)
    if any(value is not None for value in supplied_capacity_fields):
        if any(value is None for value in supplied_capacity_fields):
            raise ValueError(
                "event_times, event_ids, and capacity_per_hour are all required"
            )
        metrics["capacity"] = capacity_metrics(
            labels,
            probabilities,
            event_times,  # type: ignore[arg-type]
            event_ids,  # type: ignore[arg-type]
            capacity_per_hour=int(capacity_per_hour),
            case_ids=case_ids,
            known_case_ids=known_case_ids,
            amounts=amounts,
        )
    return metrics


def merge_streaming_evaluation(
    chunks: Iterable[Mapping[str, Sequence[Any]]],
    *,
    threshold: float,
    capacity_per_hour: int,
) -> dict[str, Any]:
    """Combine scored chunks and evaluate; intended for bounded validation/test.

    Full AMLworld validation/test prediction vectors are small enough to retain
    labels, scores, timestamps, IDs, case strings, and amounts in memory while
    feature matrices are discarded chunk-by-chunk.
    """
    fields: dict[str, list[Any]] = {
        "labels": [],
        "probabilities": [],
        "event_times": [],
        "event_ids": [],
        "case_ids": [],
        "amounts": [],
    }
    saw_cases = saw_amounts = True
    for chunk in chunks:
        for required in ("labels", "probabilities", "event_times", "event_ids"):
            if required not in chunk:
                raise ValueError(f"evaluation chunk is missing {required!r}")
            fields[required].extend(list(chunk[required]))
        if "case_ids" in chunk:
            fields["case_ids"].extend(list(chunk["case_ids"]))
        else:
            saw_cases = False
        if "amounts" in chunk:
            fields["amounts"].extend(list(chunk["amounts"]))
        else:
            saw_amounts = False
    return evaluate_predictions(
        fields["labels"],
        fields["probabilities"],
        threshold=threshold,
        event_times=fields["event_times"],
        event_ids=fields["event_ids"],
        capacity_per_hour=capacity_per_hour,
        case_ids=fields["case_ids"] if saw_cases else None,
        amounts=fields["amounts"] if saw_amounts else None,
    )


def _validated_pair(
    labels: Sequence[Any], scores: Sequence[float], *, probability: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    target = validate_binary_labels(labels)
    risk = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(target) != len(risk) or not np.isfinite(risk).all():
        raise ModelInputError("scores must be finite and match labels")
    if probability and (np.any(risk < 0.0) or np.any(risk > 1.0)):
        raise ModelInputError("probabilities must be between zero and one")
    return target, risk


def _safe_divide(numerator: float | int, denominator: float | int) -> float:
    return 0.0 if denominator == 0 else float(numerator / denominator)


def _parse_case_ids(value: Any) -> tuple[str, ...]:
    if value is None or value is pd.NA:
        return ()
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item) for item in value if str(item))
    text = str(value).strip()
    if not text or text in {"[]", "null", "<NA>"}:
        return ()
    if text.startswith("["):
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid case ID JSON: {text!r}") from error
        if not isinstance(decoded, list):
            raise ValueError("case ID JSON must be a list")
        return tuple(str(item) for item in decoded if str(item))
    return (text,)


__all__ = [
    "average_precision",
    "calibration_summary",
    "capacity_metrics",
    "choose_threshold",
    "classification_metrics",
    "evaluate_predictions",
    "merge_streaming_evaluation",
    "roc_auc",
]
