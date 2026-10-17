from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.app.ml.artifacts import load_model_bundle, verify_model_bundle
from backend.app.ml.raw_features import (
    HASH_BUCKETS,
    RAW_FEATURE_COLUMNS,
    plan_chronological_splits,
    transform_raw_chunk,
)
from backend.app.ml.training import TrainingConfig, collect_training_sample
from pipelines.parse_amlworld_patterns import transaction_fingerprint
from scripts.train_sprint2 import (
    load_configuration,
    make_toy_source,
    reject_development_source,
    train_and_package,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_checked_in_config_reaches_all_model_components():
    raw = load_configuration(PROJECT_ROOT / "config" / "model.json")
    config = TrainingConfig.from_mapping(raw)

    assert config.minimum_positives == {
        "train": 50,
        "validation": 50,
        "test": 50,
    }
    assert config.minimum_for("test") == 50
    assert set(config.categorical_features) == {
        name for name in RAW_FEATURE_COLUMNS if name.endswith("_hash")
    }
    assert config.lightgbm_parameters["categorical_feature"] == list(
        config.categorical_features
    )
    assert config.lightgbm_parameters["force_row_wise"] is True
    assert config.lightgbm_parameters["force_col_wise"] is False
    assert config.isolation_forest_parameters["number_of_estimators"] == 200
    assert config.isolation_forest_parameters["maximum_samples"] == 8192
    assert config.logistic_baseline_parameters["class_weight"] == "balanced"
    assert config.xgboost_parameters["num_boost_round"] == 1000


def test_raw_features_use_bounded_categorical_buckets_and_ground_truth_metadata():
    raw = make_toy_source(120)
    transformed = transform_raw_chunk(raw, split="train")

    categorical = transformed.features.loc[
        :, [name for name in RAW_FEATURE_COLUMNS if name.endswith("_hash")]
    ].to_numpy()
    assert np.equal(categorical, np.floor(categorical)).all()
    assert categorical.min() >= 0
    assert categorical.max() <= HASH_BUCKETS
    assert "transaction_fingerprint" not in transformed.features
    assert "transaction_amount" not in transformed.features

    positive = int(np.flatnonzero(transformed.labels == 1)[0])
    negative = int(np.flatnonzero(transformed.labels == 0)[0])
    fields = raw.iloc[positive].loc[
        [
            "Timestamp",
            "From Bank",
            "Account",
            "To Bank",
            "Account.1",
            "Amount Received",
            "Receiving Currency",
            "Amount Paid",
            "Payment Currency",
            "Payment Format",
            "Is Laundering",
        ]
    ].tolist()
    assert (
        transformed.metadata.loc[positive, "transaction_fingerprint"]
        == transaction_fingerprint(fields)
    )
    assert pd.isna(
        transformed.metadata.loc[negative, "transaction_fingerprint"]
    )


def test_negative_sampling_is_exact_and_independent_of_chunk_size():
    source = make_toy_source(240)
    boundaries = plan_chronological_splits(source, chunk_size=29)
    first = collect_training_sample(
        source,
        boundaries,
        negative_to_positive_ratio=2,
        seed=71,
        chunk_size=17,
    )
    second = collect_training_sample(
        source,
        boundaries,
        negative_to_positive_ratio=2,
        seed=71,
        chunk_size=53,
    )

    assert set(first.metadata["source_row_number"]) == set(
        second.metadata["source_row_number"]
    )
    positives = int(first.labels.sum())
    assert positives == boundaries.positive_count("train")
    assert int((first.labels == 0).sum()) == min(
        boundaries.negative_count("train"), positives * 2
    )
    assert set(first.metadata["split"]) == {"train"}


def test_chronological_plan_rejects_a_zero_positive_test_period():
    source = make_toy_source(180)
    source["Is Laundering"] = "0"
    source.loc[[1, 10, 135], "Is Laundering"] = "1"

    with pytest.raises(ValueError, match=r"test has 0 positive labels"):
        plan_chronological_splits(
            source,
            chunk_size=31,
            minimum_positives={"train": 1, "validation": 1, "test": 1},
        )


def test_dev_100000_artifact_is_rejected_with_actionable_reason(tmp_path: Path):
    run = tmp_path / "dev-100000"
    matrix = run / "model_matrix"
    matrix.mkdir(parents=True)
    pd.DataFrame({"label": ["0", "0", "0"]}).to_parquet(
        matrix / "test.parquet", index=False
    )

    with pytest.raises(ValueError, match=r"zero positive labels"):
        reject_development_source(run)


def test_quick_mode_trains_evaluates_and_loads_verified_bundle(tmp_path: Path):
    configuration = load_configuration(PROJECT_ROOT / "config" / "model.json")
    output = tmp_path / "quick-bundle"
    summary = train_and_package(
        configuration,
        output_override=output,
        quick=True,
        bundle_id="pytest-quick",
    )

    assert summary["status"] == "complete"
    assert summary["validation"]["rows"] > 0
    assert summary["validation"]["positives"] > 0
    assert summary["test"]["rows"] > 0
    assert summary["test"]["positives"] > 0
    verified = verify_model_bundle(output)
    assert {
        "primary_model",
        "calibration",
        "logistic_baseline",
        "anomaly_signal",
        "metrics",
    } <= set(verified.manifest["files"])

    calibration = json.loads((output / "calibration.json").read_text())
    metrics = json.loads((output / "metrics.json").read_text())
    assert calibration["fitted_rows"] == metrics["validation"]["rows"]
    assert metrics["threshold_selection"]["split"] == "validation"
    assert metrics["validation"]["full_split_scored"] is True
    assert metrics["test"]["full_split_scored"] is True

    loaded = load_model_bundle(output, allow_trusted_secondary_pickle=True)
    features = transform_raw_chunk(make_toy_source(120), split="test").features
    probabilities = loaded.model.predict_probability(features.iloc[:5])
    assert probabilities.shape == (5,)
    assert np.isfinite(probabilities).all()
    assert loaded.logistic_baseline is not None
    assert loaded.anomaly_signal is not None
