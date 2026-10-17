from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from backend.app.services.case_queue import QueueConfig
from pipelines.build_case_queue import write_queue_artifacts
from scripts.run_sprint1 import preflight_outputs, run_sprint1
from scripts.verify_sprint1 import (
    Sprint1VerificationError,
    layout_for,
    verify_sprint1,
    write_verification_report,
)


RAW_NAMES = ("HI-Small_Trans.csv", "HI-Small_Patterns.txt")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _time(value: object) -> str:
    return pd.Timestamp(value).isoformat()


def _build_valid_run(root: Path):
    layout = layout_for(root)
    raw = root / "data" / "raw" / "ibm_aml_hi_small"
    raw.mkdir(parents=True)
    (raw / RAW_NAMES[0]).write_text("tiny transaction fixture\n", encoding="utf-8")
    (raw / RAW_NAMES[1]).write_text("tiny pattern fixture\n", encoding="utf-8")
    records = [
        {
            "name": name,
            "sha256": _sha(raw / name),
            "expected_sha256": _sha(raw / name),
        }
        for name in RAW_NAMES
    ]
    _json(
        root / "config" / "datasets.json",
        {
            "schema_version": 1,
            "datasets": {"ibm_aml_hi_small": {"files": records}},
        },
    )
    _json(
        raw / "manifest.json",
        {
            "schema_version": 1,
            "dataset_id": "ibm_aml_hi_small",
            "files": records,
        },
    )

    times = pd.date_range("2022-09-01", periods=9, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "event_id": [f"event-{index}" for index in range(9)],
            "event_time": times,
            "source_row_number": range(2, 11),
            "label": [0, 1, 0, 0, 0, 1, 0, 1, 0],
            "pattern_matched": [False, True, False, False, False, True, False, True, False],
            "laundering_case_ids": ["[]", '["case-a"]', "[]", "[]", "[]", '["case-b"]', "[]", '["case-c"]', "[]"],
            "laundering_typologies": ["[]", '["FAN-OUT"]', "[]", "[]", "[]", '["CYCLE"]', "[]", '["STACK"]', "[]"],
            "amount": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
        }
    )
    feature_file = layout.feature_store / "event_date=2022-09-01" / "part-000000.parquet"
    feature_file.parent.mkdir(parents=True)
    frame.to_parquet(feature_file, index=False)
    hashes = {name: _sha(raw / name) for name in RAW_NAMES}
    _json(
        layout.feature_store / "manifest.json",
        {
            "schema_version": 1,
            "dataset_id": "ibm_aml_hi_small",
            "source": {"sha256": hashes, "physical_rows_scanned": 9},
            "output": {
                "rows": 9,
                "partitions": 1,
                "parquet_files": 1,
                "minimum_event_time": _time(times[0]),
                "maximum_event_time": _time(times[-1]),
            },
            "features": {"feature_columns": ["amount"]},
        },
    )

    split_records = {}
    split_frames = {}
    for name, bounds in {"train": (0, 3), "validation": (3, 6), "test": (6, 9)}.items():
        subset = frame.iloc[bounds[0] : bounds[1]].copy()
        split_frames[name] = subset
        destination = layout.splits / name / "event_date=2022-09-01" / "part-000000.parquet"
        destination.parent.mkdir(parents=True)
        subset.to_parquet(destination, index=False)
        split_records[name] = {
            "rows": len(subset),
            "minimum_event_time": _time(subset["event_time"].min()),
            "maximum_event_time": _time(subset["event_time"].max()),
        }
    _json(
        layout.splits / "manifest.json",
        {
            "schema_version": 1,
            "total_rows": 9,
            "splits": split_records,
        },
    )

    state = {"training_rows": 3, "feature_columns": ["amount"]}
    state_path = layout.model_matrix / "preprocessor.json"
    _json(state_path, state)
    matrix_records = {}
    for name, subset in split_frames.items():
        destination = layout.model_matrix / f"{name}.parquet"
        subset[["event_id", "label", "amount"]].to_parquet(destination, index=False)
        matrix_records[name] = {"rows": len(subset), "output": str(destination.resolve())}
    _json(
        layout.model_matrix / "manifest.json",
        {
            "schema_version": 1,
            "fit_split": "train",
            "fit_rows": 3,
            "state_path": str(state_path.resolve()),
            "state_sha256": _sha(state_path),
            "feature_columns": ["amount"],
            "splits": matrix_records,
        },
    )

    write_queue_artifacts(
        split_frames["test"],
        layout.case_queue,
        config=QueueConfig(capacity_per_hour=1),
    )

    layout.eda.mkdir(parents=True)
    _json(layout.eda / "summary.json", {"profile_scope": {"rows": 9}})
    for name in (
        "column_profile.csv",
        "numeric_profile.csv",
        "categorical_top_values.csv",
        "daily_activity.csv",
        "typology_profile.csv",
        "quality_checks.csv",
    ):
        (layout.eda / name).write_text("column,value\n", encoding="utf-8")
    for name in ("class_balance.png", "daily_volume.png", "amount_distribution.png"):
        (layout.eda / name).write_bytes(b"png")
    layout.data_dictionary.parent.mkdir(parents=True, exist_ok=True)
    with layout.data_dictionary.open("w", encoding="utf-8", newline="") as stream:
        columns = [
            "feature_name", "dtype", "source_table", "null_rate",
            "transformation_applied", "leakage_risk_flag",
        ]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for name, leakage in (
            ("amount", False),
            ("label", True),
            ("laundering_case_ids", True),
            ("laundering_typologies", True),
        ):
            writer.writerow(
                {
                    "feature_name": name,
                    "dtype": "float64",
                    "source_table": "fixture",
                    "null_rate": "0",
                    "transformation_applied": "fixture",
                    "leakage_risk_flag": str(leakage).lower(),
                }
            )
    return layout


def test_bounded_run_layout_is_isolated_and_validated(tmp_path: Path) -> None:
    layout = layout_for(tmp_path, "dev-1000")
    assert layout.feature_store == tmp_path / "data" / "processed" / "runs" / "dev-1000" / "feature_store"
    assert layout.data_dictionary == tmp_path / "reports" / "runs" / "dev-1000" / "data_dictionary.csv"
    with pytest.raises(ValueError):
        layout_for(tmp_path, "../escape")


def test_preflight_refuses_any_existing_output(tmp_path: Path) -> None:
    layout = layout_for(tmp_path, "rerun")
    layout.data_dictionary.parent.mkdir(parents=True)
    layout.data_dictionary.write_text("existing\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        preflight_outputs(layout)


def test_verifier_accepts_complete_consistent_run(tmp_path: Path) -> None:
    layout = _build_valid_run(tmp_path)
    report = verify_sprint1(layout)
    assert report["status"] == "pass"
    assert report["checks"]["feature_store"]["rows"] == 9
    assert report["checks"]["case_queue"]["ground_truth_isolated"] is True


def test_verifier_rejects_ground_truth_in_analyst_alerts(tmp_path: Path) -> None:
    layout = _build_valid_run(tmp_path)
    path = layout.case_queue / "analyst_alerts.parquet"
    alerts = pd.read_parquet(path)
    alerts["label"] = 1
    alerts.to_parquet(path, index=False)
    with pytest.raises(Sprint1VerificationError, match="expose ground truth"):
        verify_sprint1(layout)


def test_verification_report_is_create_only(tmp_path: Path) -> None:
    destination = tmp_path / "verification.json"
    write_verification_report({"status": "pass"}, destination)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_verification_report({"status": "pass"}, destination)


def test_runner_executes_complete_synthetic_sprint(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "raw" / "ibm_aml_hi_small"
    raw.mkdir(parents=True)
    header = (
        "Timestamp,From Bank,Account,To Bank,Account,Amount Received,"
        "Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering"
    )
    rows = []
    for hour in range(12):
        label = 1 if hour == 5 else 0
        rows.append(
            f"2022/09/01 {hour:02d}:00,001,S{hour % 3},002,D{hour % 4},"
            f"{hour + 1}.00,US Dollar,{hour + 1}.00,US Dollar,ACH,{label}"
        )
    transaction_path = raw / RAW_NAMES[0]
    transaction_path.write_text(
        header + "\n" + "\n".join(reversed(rows)) + "\n", encoding="utf-8"
    )
    pattern_path = raw / RAW_NAMES[1]
    pattern_path.write_text(
        "BEGIN LAUNDERING ATTEMPT - FAN-OUT: fixture\n"
        + rows[5]
        + "\nEND LAUNDERING ATTEMPT - FAN-OUT\n",
        encoding="utf-8",
    )
    version = "fixture-v1"
    registry_files = []
    manifest_files = []
    for path in (transaction_path, pattern_path):
        source_url = path.resolve().as_uri()
        digest = _sha(path)
        registry_files.append(
            {
                "name": path.name,
                "role": "fixture",
                "url": source_url,
                "sha256": digest,
            }
        )
        manifest_files.append(
            {
                "name": path.name,
                "role": "fixture",
                "source_url": source_url,
                "dataset_version": version,
                "license": "fixture-license",
                "bytes": path.stat().st_size,
                "fetched_at": "2026-01-01T00:00:00Z",
                "sha256": digest,
                "expected_sha256": digest,
            }
        )
    _json(
        tmp_path / "config" / "datasets.json",
        {
            "schema_version": 1,
            "datasets": {
                "ibm_aml_hi_small": {
                    "name": "fixture",
                    "description": "fixture",
                    "version": version,
                    "provider": "test",
                    "landing_page": "https://example.test",
                    "distribution_page": "https://example.test",
                    "access_type": "direct_public",
                    "credentials_required": False,
                    "license": {
                        "id": "fixture-license",
                        "url": "https://example.test/license",
                        "acknowledgement_required": False,
                    },
                    "files": registry_files,
                }
            },
        },
    )
    _json(
        raw / "manifest.json",
        {
            "schema_version": 1,
            "dataset_id": "ibm_aml_hi_small",
            "dataset_name": "fixture",
            "dataset_version": version,
            "files": manifest_files,
        },
    )
    _json(
        tmp_path / "config" / "pipeline.json",
        {
            "chunk_size": 4,
            "split_ratios": {"train": 0.7, "validation": 0.15, "test": 0.15},
            "queue": {"capacity_per_hour": 1, "study_seed": 17},
        },
    )
    _json(
        tmp_path / "config" / "study.json",
        {
            "queue_policy": {"capacity_per_hour": 1},
            "arms": [
                "data_only",
                "score_only",
                "generic_explanation",
                "reliability_gated_case_context",
            ],
        },
    )

    layout = layout_for(tmp_path, "e2e")
    manifest = run_sprint1(
        layout,
        max_rows=12,
        chunk_size=4,
        batch_size=4,
    )

    assert manifest["status"] == "complete"
    assert manifest["stage_results"]["feature_rows"] == 12
    assert manifest["stage_results"]["verification_status"] == "pass"
    assert layout.run_manifest.is_file()
    assert layout.verification_report.is_file()
