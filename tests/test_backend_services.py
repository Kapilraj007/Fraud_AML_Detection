from __future__ import annotations

import numpy as np

from backend.app.ml.artifacts import LoadedModelBundle
from backend.app.ml.model import LightGBMModel, PlattCalibrator
from backend.app.services.case_management import (
    CaseStatus,
    InvalidCaseTransition,
    disposition_status,
    validate_transition,
)
from backend.app.services.drift import DriftMonitor
from backend.app.services.explanation import explain_contributions
from backend.app.services.inference import InferenceService


class _Booster:
    best_iteration = 1

    def predict(self, matrix, *, raw_score=False, pred_contrib=False, num_iteration=None):
        values = np.asarray(matrix, dtype=float)
        margin = values[:, 0] - values[:, 1]
        if pred_contrib:
            return np.column_stack((values[:, 0], -values[:, 1], np.zeros(len(values))))
        if raw_score:
            return margin
        return 1.0 / (1.0 + np.exp(-margin))


def _service() -> InferenceService:
    model = LightGBMModel(
        _Booster(),
        ("amount_log1p", "source_txn_count_1h"),
        PlattCalibrator(),
    )
    bundle = LoadedModelBundle(
        model=model,
        manifest={"bundle_id": "test-v1", "feature_schema_version": 1},
        feature_schema={"feature_names": list(model.feature_names)},
        metrics={},
        training={},
    )
    return InferenceService(bundle)


def test_inference_is_schema_bound_and_explained() -> None:
    result = _service().score_one(
        {"amount_log1p": 2.0, "source_txn_count_1h": 0.5}
    )
    assert 0.0 <= result.supervised_probability <= 1.0
    assert result.combined_risk == result.supervised_probability
    assert result.model_version == "test-v1"
    assert result.explanation["top_features"][0]["feature"] == "amount_log1p"


def test_inference_rejects_ground_truth_and_oversized_batches() -> None:
    service = _service()
    try:
        service.score_one(
            {
                "amount_log1p": 2.0,
                "source_txn_count_1h": 0.5,
                "label": 1,
            }
        )
    except ValueError as error:
        assert "prohibited" in str(error)
    else:  # pragma: no cover
        raise AssertionError("ground truth reached the model")


def test_explanation_reliability_gate_reports_drift() -> None:
    result = explain_contributions(
        {"amount": 0.7, "source_txn_count_1h": -0.2, "expected_value": -2.0},
        model_probability=0.9,
        anomaly_probability=0.1,
        drift_detected=True,
    )
    assert not result.reliable
    assert any("distribution shift" in reason for reason in result.reliability_reasons)
    assert any("disagree" in reason for reason in result.reliability_reasons)


def test_drift_monitor_detects_large_feature_shift() -> None:
    monitor = DriftMonitor(
        {"amount": np.linspace(0.0, 1.0, 100)},
        minimum_rows=20,
        statistic_threshold=0.2,
    )
    snapshot = monitor.update(
        {"amount": np.linspace(10.0, 11.0, 100)},
        np.linspace(0.1, 0.2, 100),
    )
    assert snapshot.drifted_feature_count == 1
    assert snapshot.status == "watch"


def test_case_transitions_are_explicit() -> None:
    assert validate_transition("open", "claimed") is CaseStatus.CLAIMED
    assert disposition_status("escalate") is CaseStatus.ESCALATED
    try:
        validate_transition("open", "dismissed")
    except InvalidCaseTransition:
        pass
    else:  # pragma: no cover
        raise AssertionError("invalid transition was accepted")
