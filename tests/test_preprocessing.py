import json

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from backend.app.ml.preprocessing import (
    FeaturePreprocessor,
    MISSING_CATEGORY,
    UNKNOWN_CATEGORY,
)
from pipelines.preprocess_splits import preprocess_splits, resolve_split_paths


def _frame(amounts=(10.0, 20.0, np.nan, 40.0)):
    size = len(amounts)
    return pd.DataFrame({
        "event_id": [f"e{index}" for index in range(size)],
        "event_time": pd.date_range("2026-01-01", periods=size, freq="h", tz="UTC"),
        "label": [0, 1, 0, 1][:size],
        "amount": list(amounts),
        "source_txn_count_1h": [0, 1, np.nan, 3][:size],
        "source_account": ["a", "a", "b", None][:size],
        "destination_account": ["x", "y", "x", "z"][:size],
        "transaction_type": ["ACH", "WIRE", "ACH", None][:size],
        "currency": ["USD", "USD", "EUR", "USD"][:size],
    })


def test_fit_uses_train_only_and_transform_does_not_mutate_state():
    train = _frame()
    validation = _frame((1_000_000.0, 2_000_000.0, 3_000_000.0, 4_000_000.0))
    validation["source_account"] = "never-seen"
    validation["transaction_type"] = "CRYPTO"
    validation["label"] = 999

    processor = FeaturePreprocessor().fit(train)
    state_before = json.dumps(processor.to_dict(), sort_keys=True)
    train_before = processor.transform(train)
    validation_result = processor.transform(validation)
    state_after = json.dumps(processor.to_dict(), sort_keys=True)
    train_after = processor.transform(train)

    assert state_after == state_before
    pdt.assert_frame_equal(train_after, train_before)
    assert processor.state.amount_center == pytest.approx(22.5)
    assert validation_result["source_account__frequency"].eq(0).all()
    unknown = next(column for column in validation_result
                   if column.endswith(f"={UNKNOWN_CATEGORY}"))
    assert validation_result[unknown].eq(1).all()
    assert "label" not in processor.feature_columns
    assert all("event_id" not in column for column in processor.feature_columns)


def test_missing_indicators_frequency_encoding_one_hot_and_amount_copy():
    train = _frame()
    result = FeaturePreprocessor().fit_transform(train)

    assert result.loc[2, "amount"] == pytest.approx(20.0)
    assert result["amount__missing"].tolist() == [0, 0, 1, 0]
    assert result["source_txn_count_1h__missing"].tolist() == [0, 0, 1, 0]
    assert result["source_account__frequency"].tolist() == [0.5, 0.5, 0.25, 0.25]
    assert result["amount_scaled"].mean() == pytest.approx(0.0)
    assert result["amount_scaled"].std(ddof=0) == pytest.approx(1.0)

    ach_column = next(column for column in result if column.endswith("=value%3AACH"))
    missing_column = next(column for column in result
                          if column.startswith("transaction_type__one_hot=")
                          and column.endswith(f"={MISSING_CATEGORY}"))
    assert result[ach_column].tolist() == [1, 0, 1, 0]
    assert result[missing_column].tolist() == [0, 0, 0, 1]
    assert not result.isna().any().any()


def test_schema_is_deterministic_and_state_round_trips(tmp_path):
    train = _frame()
    processor = FeaturePreprocessor().fit(train)
    path = processor.save(tmp_path / "nested" / "preprocessor.json")
    restored = FeaturePreprocessor.load(path)

    assert restored.to_dict() == processor.to_dict()
    assert restored.feature_columns == processor.feature_columns
    pdt.assert_frame_equal(restored.transform(train), processor.transform(train))
    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert parsed["state"]["version"] == 1
    assert parsed["state"]["training_rows"] == len(train)

    shuffled = train[[*reversed(train.columns)]]
    second = FeaturePreprocessor().fit(shuffled)
    assert second.feature_columns == processor.feature_columns

    tampered = processor.to_dict()
    tampered["state"]["feature_columns"].append("label")
    with pytest.raises(ValueError, match="does not match"):
        FeaturePreprocessor.from_dict(tampered)


def test_unseen_and_missing_values_use_frozen_buckets():
    processor = FeaturePreprocessor().fit(_frame())
    future = pd.DataFrame({
        "amount": [np.inf, -np.inf],
        "source_txn_count_1h": [10, 11],
        "source_account": ["new", None],
        "destination_account": ["new", None],
        "transaction_type": ["NEW-TYPE", None],
        "currency": ["JPY", None],
    })
    result = processor.transform(future)
    assert result["amount__missing"].tolist() == [1, 1]
    assert result["source_account__frequency"].tolist() == [0.0, 0.25]
    assert result.shape[1] == len(processor.feature_columns)
    assert tuple(result.columns) == processor.feature_columns


def test_optional_ieee_device_fields_are_train_fitted_and_absent_safe():
    train = _frame()
    train["device_info"] = ["ios-17", "ios-17", "android-15", None]
    train["device_type"] = ["mobile", "mobile", "mobile", "desktop"]
    processor = FeaturePreprocessor().fit(train)

    future = _frame()
    future["device_info"] = ["unseen-device"] * len(future)
    future["device_type"] = ["console"] * len(future)
    transformed = processor.transform(future)
    assert transformed["device_info__frequency"].eq(0).all()
    device_unknown = next(
        column for column in transformed
        if column.startswith("device_type__one_hot=")
        and column.endswith(f"={UNKNOWN_CATEGORY}")
    )
    assert transformed[device_unknown].eq(1).all()

    # Device inputs are optional: a split missing them maps to the frozen
    # missing buckets rather than changing the output schema.
    without_devices = processor.transform(_frame())
    assert tuple(without_devices.columns) == processor.feature_columns
    assert without_devices["device_info__frequency"].eq(0.25).all()


def test_rejects_labels_as_explicit_numeric_features():
    with pytest.raises(ValueError, match="metadata/label"):
        FeaturePreprocessor(numeric_columns=["amount", "label"]).fit(_frame())


def test_evaluation_metadata_cannot_change_or_expand_model_features():
    clean = _frame()
    annotated = clean.copy()
    annotated["label"] = [100, 200, 300, 400]
    annotated["case_id"] = ["case-a", "case-b", "case-c", "case-d"]
    annotated["typology"] = ["fan-out", "cycle", "scatter", "fan-in"]
    annotated["pattern_id"] = [9001, 9002, 9003, 9004]
    annotated["pattern_detail"] = ["d1", "d2", "d3", "d4"]
    annotated["laundering_case_ids"] = ['["a"]', '["b"]', "[]", "[]"]
    annotated["laundering_case_count"] = [1, 1, 0, 0]
    annotated["pattern_matched"] = [True, True, False, False]
    annotated["analyst_outcome"] = ["hit", "miss", "hit", "miss"]

    clean_processor = FeaturePreprocessor().fit(clean)
    annotated_processor = FeaturePreprocessor().fit(annotated)

    assert annotated_processor.to_dict() == clean_processor.to_dict()
    pdt.assert_frame_equal(
        annotated_processor.transform(annotated),
        clean_processor.transform(clean),
    )
    forbidden_fragments = (
        "label", "case_id", "typology", "pattern", "laundering", "analyst"
    )
    assert not any(
        fragment in column
        for column in annotated_processor.feature_columns
        for fragment in forbidden_fragments
    )


def test_pipeline_fits_train_once_and_preserves_metadata(tmp_path):
    split_root = tmp_path / "splits"
    split_root.mkdir()
    train = _frame()
    validation = _frame((1000.0, 2000.0, 3000.0, 4000.0))
    test = _frame((5000.0, 6000.0, 7000.0, 8000.0))
    for frame in (train, validation, test):
        frame["case_id"] = [f"case-{index}" for index in range(len(frame))]
        frame["typology"] = "fan-out"
        frame["pattern_id"] = np.arange(len(frame))
        frame["pattern_detail"] = "fixture"
        frame["laundering_case_ids"] = '["case-0"]'
        frame["laundering_typologies"] = '["FAN-OUT"]'
        frame["laundering_case_positions"] = "[]"
        frame["laundering_case_count"] = 1
        frame["pattern_matched"] = True
    train.to_parquet(split_root / "train.parquet", index=False)
    validation.to_parquet(split_root / "validation.parquet", index=False)
    test.to_parquet(split_root / "test.parquet", index=False)

    paths = resolve_split_paths(split_root)
    output = tmp_path / "processed"
    manifest = preprocess_splits(
        paths["train"], paths["validation"], paths["test"], output
    )

    assert manifest["fit_split"] == "train"
    assert manifest["fit_rows"] == len(train)
    assert manifest["state_sha256"]
    assert (output / "preprocessor.json").is_file()
    assert (output / "manifest.json").is_file()
    validation_matrix = pd.read_parquet(output / "validation.parquet")
    assert validation_matrix["event_id"].tolist() == validation["event_id"].tolist()
    assert validation_matrix["label"].tolist() == validation["label"].tolist()
    assert validation_matrix["case_id"].tolist() == validation["case_id"].tolist()
    assert validation_matrix["typology"].tolist() == validation["typology"].tolist()
    assert validation_matrix["pattern_id"].tolist() == validation["pattern_id"].tolist()
    assert validation_matrix["pattern_detail"].tolist() == (
        validation["pattern_detail"].tolist()
    )
    assert validation_matrix["laundering_case_ids"].tolist() == (
        validation["laundering_case_ids"].tolist()
    )
    assert validation_matrix["laundering_typologies"].tolist() == (
        validation["laundering_typologies"].tolist()
    )
    assert validation_matrix["laundering_case_positions"].tolist() == (
        validation["laundering_case_positions"].tolist()
    )
    assert validation_matrix["laundering_case_count"].tolist() == (
        validation["laundering_case_count"].tolist()
    )
    assert validation_matrix["pattern_matched"].tolist() == (
        validation["pattern_matched"].tolist()
    )
    expected = FeaturePreprocessor.load(output / "preprocessor.json").transform(
        validation
    )
    pdt.assert_frame_equal(
        validation_matrix.loc[:, list(expected.columns)], expected,
        check_index_type=False,
    )
