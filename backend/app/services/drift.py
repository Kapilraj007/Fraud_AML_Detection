"""Feature and prediction drift monitoring for time-ordered score batches."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from threading import Lock
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import ks_2samp


@dataclass(frozen=True)
class FeatureDrift:
    feature: str
    statistic: float
    p_value: float
    drifted: bool
    reference_rows: int
    current_rows: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "drifted": self.drifted,
            "reference_rows": self.reference_rows,
            "current_rows": self.current_rows,
        }


@dataclass(frozen=True)
class DriftSnapshot:
    observed_at: str
    status: str
    score_drift: bool
    drifted_feature_count: int
    tested_feature_count: int
    features: tuple[FeatureDrift, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "observed_at": self.observed_at,
            "status": self.status,
            "score_drift": self.score_drift,
            "drifted_feature_count": self.drifted_feature_count,
            "tested_feature_count": self.tested_feature_count,
            "features": [feature.to_dict() for feature in self.features],
        }


class DriftMonitor:
    """KS feature checks plus a label-independent ADWIN score detector."""

    def __init__(
        self,
        reference: Mapping[str, Sequence[float]],
        *,
        p_value_threshold: float = 0.01,
        statistic_threshold: float = 0.1,
        minimum_rows: int = 30,
        adwin_delta: float = 0.002,
    ) -> None:
        if not 0 < p_value_threshold < 1:
            raise ValueError("p_value_threshold must be between zero and one")
        if not 0 <= statistic_threshold <= 1:
            raise ValueError("statistic_threshold must be between zero and one")
        if minimum_rows < 2:
            raise ValueError("minimum_rows must be at least two")
        self.reference = {
            str(name): _finite(values, f"reference {name}")
            for name, values in reference.items()
        }
        if not self.reference:
            raise ValueError("at least one reference feature is required")
        self.p_value_threshold = float(p_value_threshold)
        self.statistic_threshold = float(statistic_threshold)
        self.minimum_rows = int(minimum_rows)
        self._lock = Lock()
        try:
            from river.drift import ADWIN
        except ImportError as error:  # pragma: no cover - dependency error
            raise RuntimeError("river is required for ADWIN score monitoring") from error
        self._adwin = ADWIN(delta=float(adwin_delta))

    def update(
        self,
        current: Mapping[str, Sequence[float]],
        scores: Sequence[float],
        *,
        observed_at: datetime | None = None,
    ) -> DriftSnapshot:
        feature_results: list[FeatureDrift] = []
        for name in sorted(self.reference):
            if name not in current:
                continue
            reference_values = self.reference[name]
            current_values = _finite(current[name], f"current {name}")
            if len(reference_values) < self.minimum_rows or len(current_values) < self.minimum_rows:
                continue
            result = ks_2samp(reference_values, current_values, method="auto")
            statistic = float(result.statistic)
            p_value = float(result.pvalue)
            drifted = (
                p_value < self.p_value_threshold
                and statistic >= self.statistic_threshold
            )
            feature_results.append(
                FeatureDrift(
                    name,
                    statistic,
                    p_value,
                    drifted,
                    len(reference_values),
                    len(current_values),
                )
            )

        score_values = _finite(scores, "prediction scores")
        score_drift = False
        with self._lock:
            for value in score_values:
                self._adwin.update(float(value))
                score_drift = bool(self._adwin.drift_detected) or score_drift
        drifted_count = sum(result.drifted for result in feature_results)
        if score_drift or drifted_count >= 2:
            status = "drift_detected"
        elif drifted_count == 1:
            status = "watch"
        else:
            status = "stable"
        timestamp = (observed_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
        return DriftSnapshot(
            timestamp.isoformat(),
            status,
            score_drift,
            drifted_count,
            len(feature_results),
            tuple(feature_results),
        )


def _finite(values: Sequence[float], label: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    array = array[np.isfinite(array)]
    if not len(array):
        raise ValueError(f"{label} has no finite values")
    return array


__all__ = ["DriftMonitor", "DriftSnapshot", "FeatureDrift"]
