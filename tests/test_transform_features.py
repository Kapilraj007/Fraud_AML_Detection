"""Focused leakage and adapter tests for the shared causal feature engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.app.ml.transform_features import FEATURE_COLUMNS, transform_features


def _aml_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Timestamp": [
                "2022-09-01 00:00:00",
                "2022-09-01 00:10:00",
                "2022-09-01 00:20:00",
                "2022-09-01 02:00:00",
            ],
            "From Bank": [1, 2, 3, 1],
            "Account": ["alice", "bob", "carol", "alice"],
            "To Bank": [2, 3, 4, 5],
            "Account.1": ["bob", "carol", "dave", "erin"],
            "Amount Paid": [100.0, 90.0, 80.0, 50.0],
            "Payment Currency": ["USD"] * 4,
            "Payment Format": ["ACH"] * 4,
            "Is Laundering": [0, 1, 1, 0],
        }
    )


def _features(output: pd.DataFrame) -> pd.DataFrame:
    assert output.attrs["feature_columns"] == list(FEATURE_COLUMNS)
    assert "label" not in output.attrs["feature_columns"]
    return output.loc[:, list(FEATURE_COLUMNS)]


def test_future_perturbation_does_not_change_past_features() -> None:
    original = _aml_frame()
    perturbed = original.copy()
    perturbed.loc[3, "Amount Paid"] = 9_999_999.0
    perturbed.loc[3, "Account.1"] = "future-new-recipient"
    perturbed.loc[3, "Is Laundering"] = 1

    before = transform_features(original, "hi-small")
    after = transform_features(perturbed, "amlworld")
    pd.testing.assert_frame_equal(
        _features(before).iloc[:3].reset_index(drop=True),
        _features(after).iloc[:3].reset_index(drop=True),
    )


def test_current_and_equal_timestamp_rows_are_excluded() -> None:
    raw = pd.DataFrame(
        {
            "Timestamp": [
                "2022-01-01 00:00:00",
                "2022-01-01 00:00:00",
                "2022-01-01 00:30:00",
            ],
            "From Bank": [1, 1, 1],
            "Account": ["a", "a", "a"],
            "To Bank": [2, 3, 4],
            "Account.1": ["b", "c", "d"],
            "Amount Paid": [10.0, 20.0, 30.0],
            "Payment Format": ["ACH"] * 3,
            "Is Laundering": [0, 0, 0],
        }
    )
    output = transform_features(raw, "amlworld")

    assert output.loc[0, "source_txn_count_1h"] == 0
    assert output.loc[1, "source_txn_count_1h"] == 0
    assert output.loc[2, "source_txn_count_1h"] == 2
    assert output.loc[2, "source_txn_sum_1h"] == pytest.approx(30.0)
    assert output.loc[2, "source_amount_mean_30d"] == pytest.approx(15.0)
    assert output.loc[2, "amount_to_source_mean_30d_ratio"] == pytest.approx(2.0)
    assert output.loc[2, "source_unique_counterparties_30d"] == 2
    assert output.loc[2, "is_new_counterparty"] == 1


def test_chunked_processing_equals_single_pass_even_at_tie_boundary() -> None:
    raw = _aml_frame()
    tied = raw.iloc[[2]].copy()
    tied.loc[:, "From Bank"] = 9
    tied.loc[:, "Account"] = "tied"
    tied.loc[:, "To Bank"] = 1
    tied.loc[:, "Account.1"] = "alice"
    raw = pd.concat([raw.iloc[:3], tied, raw.iloc[3:]], ignore_index=True)

    whole = transform_features(raw, "amlworld")
    state: dict = {}
    first = transform_features(raw.iloc[:3], "amlworld", state=state)
    second = transform_features(raw.iloc[3:], "amlworld", state=state)
    chunked = pd.concat([first, second], ignore_index=True)

    pd.testing.assert_frame_equal(
        whole.reset_index(drop=True), chunked.reset_index(drop=True),
        check_dtype=True,
    )
    assert state["last_processed_time"] is not None


@pytest.mark.parametrize("dataset,label_column", [("amlworld", "Is Laundering")])
def test_laundering_label_is_metadata_not_a_feature(
    dataset: str, label_column: str
) -> None:
    raw = _aml_frame()
    flipped = raw.copy()
    flipped[label_column] = 1 - flipped[label_column]

    left = transform_features(raw, dataset)
    right = transform_features(flipped, dataset)
    pd.testing.assert_frame_equal(_features(left), _features(right))
    assert left["label"].tolist() != right["label"].tolist()


def test_paysim_pattern_and_disallowed_columns_are_label_independent() -> None:
    raw = pd.DataFrame(
        {
            "step": [1, 2, 3],
            "type": ["TRANSFER", "CASH_OUT", "PAYMENT"],
            "amount": [500.0, 450.0, 10.0],
            "nameOrig": ["sender", "mule", "shopper"],
            "nameDest": ["mule", "cash-agent", "merchant"],
            "oldbalanceOrg": [500.0, 999.0, 10.0],
            "newbalanceOrig": [0.0, 123.0, 0.0],
            "isFlaggedFraud": [0, 1, 1],
            "isFraud": [1, 1, 0],
        }
    )
    changed = raw.copy()
    changed["isFraud"] = [0, 0, 1]
    changed["isFlaggedFraud"] = [1, 0, 0]
    changed["oldbalanceOrg"] = [1e12, -50, 777]
    changed["newbalanceOrig"] = [-5, 1e15, 42]

    output = transform_features(raw, "PaySim")
    changed_output = transform_features(changed, "PaySim")
    pd.testing.assert_frame_equal(_features(output), _features(changed_output))
    cashout = output.iloc[1]
    assert cashout["paysim_prior_transfer_in_count_24h"] == 1
    assert cashout["paysim_prior_transfer_in_sum_24h"] == pytest.approx(500.0)
    assert cashout["paysim_transfer_to_cashout_flag"] == 1
    assert cashout["paysim_transfer_to_cashout_amount_ratio"] == pytest.approx(0.9)
    assert not any("balance" in column.casefold() for column in output.columns)
    assert "isFlaggedFraud" not in output.columns


def test_ieee_adapter_keeps_relative_time_and_documents_pseudo_entity() -> None:
    raw = pd.DataFrame(
        {
            "TransactionID": [10, 11, 12],
            "TransactionDT": [100.0, 160.0, np.nan],
            "TransactionAmt": [25.0, 30.0, np.nan],
            "ProductCD": ["W", "W", None],
            "card1": [1234, 1234, np.nan],
            "addr1": [100, 100, np.nan],
            "P_emaildomain": ["example.com", "example.com", None],
            "DeviceType": ["desktop", "desktop", None],
            "DeviceInfo": ["Windows/Chrome", "Windows/Chrome", None],
            "isFraud": [0, 1, 1],
        }
    )
    output = transform_features(raw, "IEEE-CIS")

    assert output.loc[0, "relative_time_seconds"] == pytest.approx(100.0)
    assert output.loc[1, "source_txn_count_1h"] == 1
    assert output["event_time"].isna().all()
    assert output["time_semantics"].eq("relative_seconds_unknown_origin").all()
    assert output.loc[0, "source_account"].startswith("ieee-pseudo:")
    assert output.loc[2, "source_account_missing"] == 1
    assert output.loc[2, "amount_missing"] == 1
    assert output.loc[2, "time_missing"] == 1
    assert output.loc[0, "device_type"] == "desktop"
    assert output.loc[0, "device_info"] == "Windows/Chrome"
    assert output.loc[0, "new_device_flag"] == 1
    assert output.loc[1, "new_device_flag"] == 0
    assert output.loc[:, AML_FEATURES_FOR_TEST()].to_numpy().sum() == pytest.approx(0.0)
    assert "pseudo entity" in output.attrs["entity_semantics"]
    assert "no calendar origin" in output.attrs["time_semantics"]


def AML_FEATURES_FOR_TEST() -> list[str]:
    return [
        "source_recent_inbound_count", "source_recent_inbound_sum",
        "destination_recent_inbound_count", "destination_recent_inbound_sum",
        "source_fan_out_24h", "destination_fan_in_24h",
        "rapid_pass_through_ratio", "causal_chain_depth",
    ]


def test_out_of_order_chunk_is_rejected_before_state_can_leak_future() -> None:
    raw = _aml_frame()
    state: dict = {}
    transform_features(raw.iloc[2:], "amlworld", state=state)
    with pytest.raises(ValueError, match="non-decreasing"):
        transform_features(raw.iloc[:2], "amlworld", state=state)
