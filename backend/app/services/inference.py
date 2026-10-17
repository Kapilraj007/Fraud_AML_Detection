"""Thread-safe loading and scoring of verified Sprint 2 model bundles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Mapping, Sequence

import pandas as pd

from backend.app.ml.artifacts import LoadedModelBundle, load_model_bundle
from backend.app.services.explanation import explain_contributions


PROHIBITED_INPUT_FIELDS = frozenset(
    {
        "label",
        "target",
        "is_fraud",
        "fraud_label",
        "is_laundering",
        "laundering_label",
        "case_id",
        "case_ids",
        "laundering_case_ids",
        "typology",
        "typologies",
        "pattern_id",
        "pattern_matched",
        "analyst_outcome",
        "disposition",
        "ground_truth",
        "resolved_label",
    }
)

_PROHIBITED_NORMALIZED = frozenset(
    "".join(character for character in name.casefold() if character.isalnum())
    for name in PROHIBITED_INPUT_FIELDS
)


def prohibited_input_fields(values: Mapping[str, Any]) -> set[str]:
    """Return input keys that could expose evaluation truth to inference."""
    return {
        str(name)
        for name in values
        if "".join(
            character for character in str(name).casefold() if character.isalnum()
        ) in _PROHIBITED_NORMALIZED
    }


class ModelUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class ScoreResult:
    supervised_probability: float
    anomaly_probability: float | None
    combined_risk: float
    raw_margin: float
    model_version: str
    feature_schema_version: int
    explanation: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "supervised_probability": self.supervised_probability,
            "anomaly_probability": self.anomaly_probability,
            "combined_risk": self.combined_risk,
            "raw_margin": self.raw_margin,
            "model_version": self.model_version,
            "feature_schema_version": self.feature_schema_version,
            "explanation": self.explanation,
        }


class InferenceService:
    """Own a verified immutable model bundle and enforce its input contract."""

    def __init__(
        self,
        bundle: LoadedModelBundle | None = None,
        *,
        supervised_weight: float = 0.85,
        maximum_batch_size: int = 256,
    ) -> None:
        if not 0 <= supervised_weight <= 1:
            raise ValueError("supervised_weight must be between zero and one")
        if maximum_batch_size < 1:
            raise ValueError("maximum_batch_size must be positive")
        self._bundle = bundle
        self.supervised_weight = float(supervised_weight)
        self.maximum_batch_size = int(maximum_batch_size)
        self._lock = RLock()

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        allow_trusted_secondary_pickle: bool = True,
        supervised_weight: float = 0.85,
        maximum_batch_size: int = 256,
    ) -> "InferenceService":
        bundle = load_model_bundle(
            path,
            allow_trusted_secondary_pickle=allow_trusted_secondary_pickle,
        )
        return cls(
            bundle,
            supervised_weight=supervised_weight,
            maximum_batch_size=maximum_batch_size,
        )

    @property
    def available(self) -> bool:
        return self._bundle is not None

    @property
    def model_version(self) -> str | None:
        if self._bundle is None:
            return None
        return str(self._bundle.manifest.get("bundle_id", "unknown"))

    @property
    def feature_names(self) -> tuple[str, ...]:
        if self._bundle is None:
            return ()
        return self._bundle.model.feature_names

    def score_one(
        self,
        features: Mapping[str, float],
        *,
        drift_detected: bool = False,
    ) -> ScoreResult:
        return self.score_batch([features], drift_detected=drift_detected)[0]

    def score_batch(
        self,
        rows: Sequence[Mapping[str, float]],
        *,
        drift_detected: bool = False,
    ) -> list[ScoreResult]:
        if not rows:
            raise ValueError("at least one prediction row is required")
        if len(rows) > self.maximum_batch_size:
            raise ValueError(
                f"batch size {len(rows)} exceeds maximum {self.maximum_batch_size}"
            )
        for row in rows:
            forbidden = prohibited_input_fields(row)
            if forbidden:
                raise ValueError(f"prediction input contains prohibited fields: {sorted(forbidden)}")
        with self._lock:
            bundle = self._bundle
            if bundle is None:
                raise ModelUnavailableError("no verified model bundle is loaded")
            frame = pd.DataFrame(rows)
            margins = bundle.model.predict_margin(frame)
            supervised = bundle.model.predict_probability(frame)
            contributions = bundle.model.predict_contributions(frame)
            anomaly = (
                bundle.anomaly_signal.predict(frame)
                if bundle.anomaly_signal is not None
                else None
            )
            version = str(bundle.manifest.get("bundle_id", "unknown"))
            schema_version = int(bundle.manifest.get("feature_schema_version", 1))
            answers: list[ScoreResult] = []
            for index in range(len(frame)):
                anomaly_value = None if anomaly is None else float(anomaly[index])
                supervised_value = float(supervised[index])
                combined = (
                    supervised_value
                    if anomaly_value is None
                    else self.supervised_weight * supervised_value
                    + (1.0 - self.supervised_weight) * anomaly_value
                )
                explanation = explain_contributions(
                    contributions.iloc[index].to_dict(),
                    model_probability=supervised_value,
                    anomaly_probability=anomaly_value,
                    drift_detected=drift_detected,
                ).to_dict()
                answers.append(
                    ScoreResult(
                        supervised_value,
                        anomaly_value,
                        float(combined),
                        float(margins[index]),
                        version,
                        schema_version,
                        explanation,
                    )
                )
            return answers


__all__ = [
    "InferenceService",
    "ModelUnavailableError",
    "PROHIBITED_INPUT_FIELDS",
    "ScoreResult",
    "prohibited_input_fields",
]
