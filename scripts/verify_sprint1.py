#!/usr/bin/env python3
"""Independently verify the persisted Sprint 1 acceptance contract.

The checks deliberately read manifests *and* the artifacts they describe.  A
successful pipeline exit is not accepted as evidence that rows were preserved,
splits are disjoint, or reviewer data is label-safe.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.case_queue import (  # noqa: E402
    FORBIDDEN_SCORE_NAMES,
    SEALED_METADATA_NAMES,
)


DATASET_ID = "ibm_aml_hi_small"
REQUIRED_RAW_FILES = ("HI-Small_Trans.csv", "HI-Small_Patterns.txt")
SPLIT_NAMES = ("train", "validation", "test")
REQUIRED_DICTIONARY_COLUMNS = {
    "feature_name",
    "dtype",
    "source_table",
    "null_rate",
    "transformation_applied",
    "leakage_risk_flag",
}
RUN_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")


class Sprint1VerificationError(RuntimeError):
    """Raised when a persisted Sprint 1 acceptance invariant fails."""


@dataclass(frozen=True)
class ArtifactLayout:
    """Canonical locations for either the publication run or an isolated run."""

    project_root: Path
    feature_store: Path
    splits: Path
    model_matrix: Path
    case_queue: Path
    eda: Path
    data_dictionary: Path
    run_manifest: Path
    verification_report: Path


@dataclass(frozen=True)
class _Scan:
    rows: int
    minimum: pd.Timestamp
    maximum: pd.Timestamp
    schema: pa.Schema
    source_rows: np.ndarray


def layout_for(project_root: str | Path = PROJECT_ROOT, run_name: str | None = None) -> ArtifactLayout:
    """Resolve canonical artifact paths without touching the filesystem."""
    root = Path(project_root).resolve()
    if run_name is None:
        processed = root / "data" / "processed"
        report_base = root / "reports"
        return ArtifactLayout(
            project_root=root,
            feature_store=processed / "feature_store",
            splits=processed / "splits",
            model_matrix=processed / "model_matrix",
            case_queue=processed / "case_queue",
            eda=report_base / "eda",
            data_dictionary=root / "docs" / "data_dictionary.csv",
            run_manifest=report_base / "sprint1_run.json",
            verification_report=report_base / "sprint1_verification.json",
        )
    if RUN_NAME_PATTERN.fullmatch(run_name) is None or run_name in {".", ".."}:
        raise ValueError(
            "run_name must be 1-64 safe filename characters and begin with a letter or digit"
        )
    processed = root / "data" / "processed" / "runs" / run_name
    report_base = root / "reports" / "runs" / run_name
    return ArtifactLayout(
        project_root=root,
        feature_store=processed / "feature_store",
        splits=processed / "splits",
        model_matrix=processed / "model_matrix",
        case_queue=processed / "case_queue",
        eda=report_base / "eda",
        data_dictionary=report_base / "data_dictionary.csv",
        run_manifest=report_base / "run_manifest.json",
        verification_report=report_base / "verification.json",
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sprint1(layout: ArtifactLayout) -> dict[str, Any]:
    """Run every acceptance check and return a JSON-safe report.

    The function is read-only.  It raises :class:`Sprint1VerificationError` at
    the first violated invariant so callers cannot mistake a partial audit for
    a pass.
    """
    raw = _verify_raw(layout.project_root)
    feature, feature_scan = _verify_feature_store(layout, raw)
    splits, split_scans = _verify_splits(layout, feature_scan)
    matrices = _verify_model_matrix(layout, split_scans)
    queue = _verify_queue(layout, split_scans["test"])
    reports = _verify_reports(layout, feature_scan.rows, feature["features"]["feature_columns"])
    return {
        "schema_version": 1,
        "status": "pass",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": _run_name(layout),
        "checks": {
            "raw": raw,
            "feature_store": {
                "rows": feature_scan.rows,
                "minimum_event_time": feature_scan.minimum.isoformat(),
                "maximum_event_time": feature_scan.maximum.isoformat(),
                "parquet_files": feature["output"]["parquet_files"],
            },
            "splits": splits,
            "model_matrix": matrices,
            "case_queue": queue,
            "reports": reports,
        },
    }


def write_verification_report(report: Mapping[str, Any], path: str | Path) -> Path:
    """Create, but never replace, a verification report."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite verification report: {destination}") from exc
    return destination


def _run_name(layout: ArtifactLayout) -> str | None:
    parts = layout.feature_store.parts
    try:
        index = parts.index("runs")
    except ValueError:
        return None
    return parts[index + 1] if index + 1 < len(parts) else None


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise Sprint1VerificationError(f"missing {description}: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise Sprint1VerificationError(f"invalid {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise Sprint1VerificationError(f"{description} must be a JSON object: {path}")
    return value


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise Sprint1VerificationError(message)


def _verify_raw(root: Path) -> dict[str, Any]:
    registry_path = root / "config" / "datasets.json"
    raw_root = root / "data" / "raw" / DATASET_ID
    manifest_path = raw_root / "manifest.json"
    registry = _load_json(registry_path, "dataset registry")
    manifest = _load_json(manifest_path, "raw-data manifest")
    _expect(registry.get("schema_version") == 1, "dataset registry schema_version must be 1")
    _expect(manifest.get("schema_version") == 1, "raw manifest schema_version must be 1")
    _expect(manifest.get("dataset_id") == DATASET_ID, "raw manifest dataset_id is incorrect")
    try:
        registered_files = registry["datasets"][DATASET_ID]["files"]
        manifest_files = manifest["files"]
    except (KeyError, TypeError) as exc:
        raise Sprint1VerificationError("registry/raw manifest is missing its files contract") from exc
    _expect(isinstance(registered_files, list), "registry files must be a list")
    _expect(isinstance(manifest_files, list), "raw manifest files must be a list")
    registered = _records_by_name(registered_files, "registry")
    installed = _records_by_name(manifest_files, "raw manifest")
    observed: dict[str, str] = {}
    for name in REQUIRED_RAW_FILES:
        _expect(name in registered, f"required raw file is not registered: {name}")
        _expect(name in installed, f"required raw file is absent from provenance manifest: {name}")
    # Verify every installed artifact, including an optional account table when present.
    for name, record in installed.items():
        _expect(Path(name).name == name, f"unsafe raw manifest filename: {name!r}")
        path = raw_root / name
        _expect(path.is_file(), f"manifested raw file is missing: {path}")
        digest = sha256_file(path)
        _expect(str(record.get("sha256", "")).casefold() == digest, f"raw manifest hash mismatch: {name}")
        registered_digest = registered.get(name, {}).get("sha256")
        if registered_digest:
            _expect(str(registered_digest).casefold() == digest, f"registry hash mismatch: {name}")
        expected = record.get("expected_sha256")
        if expected:
            _expect(str(expected).casefold() == digest, f"raw expected hash mismatch: {name}")
        observed[name] = digest
    return {
        "dataset_id": DATASET_ID,
        "manifest": str(manifest_path),
        "verified_files": observed,
    }


def _records_by_name(records: Iterable[Any], description: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in records:
        _expect(isinstance(raw, dict), f"{description} file records must be objects")
        name = raw.get("name")
        _expect(isinstance(name, str) and bool(name), f"{description} file record has no name")
        _expect(name not in result, f"duplicate {description} file record: {name}")
        result[name] = raw
    return result


def _parquet_files(path: Path, pattern: str = "*.parquet") -> list[Path]:
    _expect(path.is_dir(), f"artifact directory is missing: {path}")
    files = sorted(candidate for candidate in path.rglob(pattern) if candidate.is_file())
    _expect(bool(files), f"no Parquet files found under {path}")
    return files


def _verify_feature_store(
    layout: ArtifactLayout, raw: Mapping[str, Any]
) -> tuple[dict[str, Any], _Scan]:
    manifest = _load_json(layout.feature_store / "manifest.json", "feature-store manifest")
    _expect(manifest.get("schema_version") == 1, "feature manifest schema_version must be 1")
    _expect(manifest.get("dataset_id") == DATASET_ID, "feature manifest dataset_id is incorrect")
    try:
        source_hashes = manifest["source"]["sha256"]
        output = manifest["output"]
        features = manifest["features"]["feature_columns"]
    except (KeyError, TypeError) as exc:
        raise Sprint1VerificationError("feature manifest is missing its source/output/schema contract") from exc
    _expect(isinstance(features, list) and bool(features), "feature_columns must be a non-empty list")
    for name in REQUIRED_RAW_FILES:
        _expect(
            source_hashes.get(name) == raw["verified_files"].get(name),
            f"feature source hash does not match verified raw bytes: {name}",
        )
    files = _parquet_files(layout.feature_store)
    _expect(len(files) == int(output["parquet_files"]), "feature Parquet file count disagrees with manifest")
    physical_rows = int(manifest["source"]["physical_rows_scanned"])
    scan = _scan_chronological(files, physical_rows, "feature store")
    _expect(scan.rows == int(output["rows"]), "feature row count disagrees with manifest")
    _expect(scan.rows <= physical_rows, "feature output exceeds scanned raw rows")
    _expect(_same_time(scan.minimum, output.get("minimum_event_time")), "feature minimum time disagrees with manifest")
    _expect(_same_time(scan.maximum, output.get("maximum_event_time")), "feature maximum time disagrees with manifest")
    partitions = {file.parent.name for file in files}
    _expect(len(partitions) == int(output["partitions"]), "feature partition count disagrees with manifest")
    names = set(scan.schema.names)
    required = {
        "event_id", "event_time", "source_row_number", "label",
        "pattern_matched", "laundering_case_ids", "laundering_typologies",
        *map(str, features),
    }
    missing = sorted(required - names)
    _expect(not missing, f"feature schema is missing columns: {missing}")
    _expect(len(scan.schema.names) == len(names), "feature schema contains duplicate column names")
    return manifest, scan


def _scan_chronological(files: Sequence[Path], physical_rows: int, description: str) -> _Scan:
    _expect(physical_rows >= 1, f"{description} physical row bound must be positive")
    seen = np.zeros(physical_rows + 2, dtype=np.bool_)
    total = 0
    minimum: pd.Timestamp | None = None
    maximum: pd.Timestamp | None = None
    prior_time: int | None = None
    prior_row: int | None = None
    canonical_schema: pa.Schema | None = None
    for path in files:
        parquet = pq.ParquetFile(path)
        schema = parquet.schema_arrow.remove_metadata()
        if canonical_schema is None:
            canonical_schema = schema
        else:
            _expect(schema.equals(canonical_schema), f"inconsistent Parquet schema at {path}")
        _expect("event_time" in schema.names, f"event_time missing from {path}")
        _expect("source_row_number" in schema.names, f"source_row_number missing from {path}")
        partition_date = _partition_date(path)
        for batch in parquet.iter_batches(
            batch_size=100_000, columns=["event_time", "source_row_number"]
        ):
            frame = batch.to_pandas()
            times = pd.to_datetime(frame["event_time"], errors="coerce", utc=True)
            rows = pd.to_numeric(frame["source_row_number"], errors="coerce")
            _expect(not times.isna().any(), f"invalid event_time in {path}")
            _expect(not rows.isna().any(), f"invalid source_row_number in {path}")
            row_values = rows.to_numpy(dtype=np.int64)
            _expect(np.array_equal(row_values, rows.to_numpy(dtype=float)), f"non-integral source row in {path}")
            _expect(bool(np.all((row_values >= 2) & (row_values <= physical_rows + 1))), f"source row outside raw bounds in {path}")
            _expect(not bool(seen[row_values].any()), f"duplicate source row detected in {description}")
            seen[row_values] = True
            time_values = times.astype("int64").to_numpy()
            if len(time_values) > 1:
                backwards = (time_values[1:] < time_values[:-1]) | (
                    (time_values[1:] == time_values[:-1]) & (row_values[1:] < row_values[:-1])
                )
                _expect(not bool(backwards.any()), f"rows are not chronological in {path}")
            if prior_time is not None and len(time_values):
                _expect(
                    (int(time_values[0]), int(row_values[0])) >= (prior_time, int(prior_row)),
                    f"global chronology regresses at {path}",
                )
            if len(time_values):
                prior_time, prior_row = int(time_values[-1]), int(row_values[-1])
                part_min, part_max = times.min(), times.max()
                minimum = part_min if minimum is None else min(minimum, part_min)
                maximum = part_max if maximum is None else max(maximum, part_max)
                if partition_date is not None:
                    dates = times.dt.strftime("%Y-%m-%d")
                    _expect(bool(dates.eq(partition_date).all()), f"event_date partition mismatch in {path}")
            total += len(frame)
    _expect(total > 0 and minimum is not None and maximum is not None, f"{description} is empty")
    assert canonical_schema is not None
    return _Scan(total, minimum, maximum, canonical_schema, seen)


def _partition_date(path: Path) -> str | None:
    for parent in path.parents:
        if parent.name.startswith("event_date="):
            value = parent.name.split("=", 1)[1]
            return value or None
    return None


def _same_time(observed: pd.Timestamp, expected: Any) -> bool:
    if expected is None:
        return False
    try:
        return observed == pd.Timestamp(expected).tz_convert("UTC")
    except (TypeError, ValueError):
        try:
            return observed == pd.Timestamp(expected, tz="UTC")
        except (TypeError, ValueError):
            return False


def _verify_splits(layout: ArtifactLayout, feature: _Scan) -> tuple[dict[str, Any], dict[str, _Scan]]:
    manifest = _load_json(layout.splits / "manifest.json", "split manifest")
    _expect(manifest.get("schema_version") == 1, "split manifest schema_version must be 1")
    _expect(int(manifest.get("total_rows", -1)) == feature.rows, "split manifest total does not match features")
    scans: dict[str, _Scan] = {}
    ownership = np.zeros(len(feature.source_rows), dtype=np.uint8)
    split_report: dict[str, Any] = {}
    for owner, name in enumerate(SPLIT_NAMES, start=1):
        files = _parquet_files(layout.splits / name)
        scan = _scan_chronological(files, len(feature.source_rows) - 2, f"{name} split")
        _expect(scan.schema.equals(feature.schema), f"{name} schema differs from feature store")
        current = scan.source_rows
        _expect(not bool((current & ~feature.source_rows).any()), f"{name} contains rows absent from feature store")
        _expect(not bool((ownership[current] != 0).any()), f"source rows overlap across splits at {name}")
        ownership[current] = owner
        record = manifest.get("splits", {}).get(name, {})
        _expect(scan.rows == int(record.get("rows", -1)), f"{name} row count disagrees with manifest")
        _expect(_same_time(scan.minimum, record.get("minimum_event_time")), f"{name} minimum time disagrees with manifest")
        _expect(_same_time(scan.maximum, record.get("maximum_event_time")), f"{name} maximum time disagrees with manifest")
        scans[name] = scan
        split_report[name] = {
            "rows": scan.rows,
            "minimum_event_time": scan.minimum.isoformat(),
            "maximum_event_time": scan.maximum.isoformat(),
        }
    _expect(sum(scan.rows for scan in scans.values()) == feature.rows, "split rows do not preserve feature rows")
    _expect(bool(np.array_equal(ownership > 0, feature.source_rows)), "split membership does not exactly cover feature rows")
    _expect(scans["train"].maximum < scans["validation"].minimum, "train/validation time ranges overlap")
    _expect(scans["validation"].maximum < scans["test"].minimum, "validation/test time ranges overlap")
    return split_report, scans


def _verify_model_matrix(layout: ArtifactLayout, splits: Mapping[str, _Scan]) -> dict[str, Any]:
    manifest = _load_json(layout.model_matrix / "manifest.json", "model-matrix manifest")
    _expect(manifest.get("schema_version") == 1, "model manifest schema_version must be 1")
    _expect(manifest.get("fit_split") == "train", "preprocessor was not fitted on train")
    _expect(int(manifest.get("fit_rows", -1)) == splits["train"].rows, "preprocessor fit row count is wrong")
    feature_columns = manifest.get("feature_columns")
    _expect(isinstance(feature_columns, list) and bool(feature_columns), "model feature_columns must be non-empty")
    _expect(len(feature_columns) == len(set(feature_columns)), "model feature_columns contains duplicates")
    forbidden = FORBIDDEN_SCORE_NAMES | SEALED_METADATA_NAMES
    leaked = sorted(column for column in feature_columns if _normalise(column) in forbidden)
    _expect(not leaked, f"ground-truth fields entered model features: {leaked}")
    state_path = Path(str(manifest.get("state_path", "")))
    if not state_path.is_absolute():
        state_path = layout.project_root / state_path
    _expect(state_path.is_file(), f"preprocessor state is missing: {state_path}")
    _expect(sha256_file(state_path) == manifest.get("state_sha256"), "preprocessor state hash mismatch")
    state_document = _load_json(state_path, "preprocessor state")
    state = state_document.get("state", state_document)
    _expect(isinstance(state, dict), "preprocessor state payload must be an object")
    _expect(int(state.get("training_rows", -1)) == splits["train"].rows, "preprocessor state training_rows is wrong")
    _expect(state.get("feature_columns") == feature_columns, "preprocessor state schema differs from manifest")
    common_schema: pa.Schema | None = None
    result: dict[str, Any] = {"fit_split": "train", "feature_count": len(feature_columns)}
    for name in SPLIT_NAMES:
        record = manifest.get("splits", {}).get(name, {})
        output = Path(str(record.get("output", layout.model_matrix / f"{name}.parquet")))
        if not output.is_absolute():
            output = layout.project_root / output
        _expect(output.is_file(), f"model matrix is missing: {output}")
        parquet = pq.ParquetFile(output)
        schema = parquet.schema_arrow.remove_metadata()
        _expect(parquet.metadata.num_rows == splits[name].rows, f"{name} matrix row count is wrong")
        _expect(int(record.get("rows", -1)) == splits[name].rows, f"{name} matrix manifest row count is wrong")
        _expect(set(feature_columns).issubset(schema.names), f"{name} matrix is missing model features")
        projected = pa.schema([schema.field(column) for column in feature_columns])
        if common_schema is None:
            common_schema = projected
        else:
            _expect(projected.equals(common_schema), f"{name} model feature schema differs")
        result[f"{name}_rows"] = splits[name].rows
    return result


def _verify_queue(layout: ArtifactLayout, test: _Scan) -> dict[str, Any]:
    manifest = _load_json(layout.case_queue / "manifest.json", "case-queue manifest")
    _expect(manifest.get("schema_version") == 1, "queue manifest schema_version must be 1")
    outputs = manifest.get("outputs", {})
    alerts_path = _safe_child(layout.case_queue, outputs.get("analyst_alerts"), "analyst alerts")
    key_path = _safe_child(layout.case_queue, outputs.get("sealed_evaluation_key"), "evaluation key")
    _expect(alerts_path.resolve() != key_path.resolve(), "analyst alerts and evaluation key are not separate")
    alerts = pd.read_parquet(alerts_path)
    alert_schema = set(alerts.columns)
    forbidden = FORBIDDEN_SCORE_NAMES | SEALED_METADATA_NAMES
    leaked = sorted(column for column in alert_schema if _normalise(column) in forbidden)
    _expect(not leaked, f"reviewer-facing alerts expose ground truth: {leaked}")
    required_alerts = {"review_id", "event_id", "alert_hour", "risk_score", "score_source"}
    _expect(required_alerts.issubset(alert_schema), "reviewer alerts are missing queue fields")
    _expect(not alerts["review_id"].astype(str).duplicated().any(), "review_id is duplicated in alerts")
    score_sources = {_normalise(value) for value in alerts["score_source"].dropna().astype(str)}
    _expect(not (score_sources & forbidden), "queue score_source names a ground-truth field")
    capacity = int(manifest.get("queue_config", {}).get("capacity_per_hour", 0))
    _expect(capacity >= 1, "queue capacity is invalid")
    hourly = alerts.groupby("alert_hour", dropna=False).size()
    _expect(hourly.empty or int(hourly.max()) <= capacity, "analyst queue exceeds hourly capacity")

    key = pq.ParquetFile(key_path)
    _expect("label" in key.schema_arrow.names, "sealed evaluation key has no label")
    _expect(int(key.metadata.num_rows) == test.rows, "evaluation key does not cover the test split")
    selected_ids: set[str] = set()
    for batch in key.iter_batches(columns=["review_id", "selected_by_capacity"]):
        frame = batch.to_pandas()
        chosen = frame.loc[frame["selected_by_capacity"].astype(bool), "review_id"].astype(str)
        _expect(not bool(set(chosen) & selected_ids), "selected review_id is duplicated in evaluation key")
        selected_ids.update(chosen)
    alert_ids = set(alerts["review_id"].astype(str))
    _expect(alert_ids == selected_ids, "alerts do not exactly match selected evaluation-key rows")
    _expect(int(manifest.get("input_rows", -1)) == test.rows, "queue input row count is wrong")
    _expect(int(manifest.get("reviewer_alert_rows", -1)) == len(alerts), "queue alert row count is wrong")
    _expect(int(manifest.get("evaluation_key_rows", -1)) == test.rows, "queue key row count is wrong")
    return {
        "input_rows": test.rows,
        "reviewer_alert_rows": len(alerts),
        "capacity_per_hour": capacity,
        "ground_truth_isolated": True,
    }


def _safe_child(root: Path, value: Any, description: str) -> Path:
    _expect(isinstance(value, str) and Path(value).name == value, f"unsafe {description} filename")
    path = root / value
    _expect(path.is_file(), f"missing {description}: {path}")
    return path


def _verify_reports(
    layout: ArtifactLayout, feature_rows: int, feature_columns: Sequence[str]
) -> dict[str, Any]:
    summary = _load_json(layout.eda / "summary.json", "EDA summary")
    scope = summary.get("profile_scope", {})
    _expect(int(scope.get("rows", -1)) == feature_rows, "EDA row scope does not match feature store")
    required_eda = {
        "column_profile.csv", "numeric_profile.csv", "categorical_top_values.csv",
        "daily_activity.csv", "typology_profile.csv", "quality_checks.csv",
        "class_balance.png", "daily_volume.png", "amount_distribution.png",
    }
    missing_eda = sorted(name for name in required_eda if not (layout.eda / name).is_file())
    _expect(not missing_eda, f"EDA artifacts are missing: {missing_eda}")
    _expect(layout.data_dictionary.is_file(), f"data dictionary is missing: {layout.data_dictionary}")
    with layout.data_dictionary.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        _expect(reader.fieldnames is not None, "data dictionary has no header")
        _expect(REQUIRED_DICTIONARY_COLUMNS.issubset(reader.fieldnames), "data dictionary header is incomplete")
        rows = list(reader)
    by_name = {row.get("feature_name", ""): row for row in rows}
    missing_features = sorted(set(map(str, feature_columns)) - set(by_name))
    _expect(not missing_features, f"data dictionary is missing features: {missing_features}")
    for name in ("label", "laundering_case_ids", "laundering_typologies"):
        if name in by_name:
            _expect(
                str(by_name[name].get("leakage_risk_flag", "")).casefold() == "true",
                f"data dictionary does not flag leakage metadata: {name}",
            )
    return {
        "eda_rows": feature_rows,
        "data_dictionary_entries": len(rows),
        "required_eda_artifacts": len(required_eda) + 1,
    }


def _normalise(value: object) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--run-name")
    parser.add_argument(
        "--output",
        type=Path,
        help="Create a JSON verification report; defaults to the selected run's canonical path",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print the audit result without creating a report file",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    layout = layout_for(args.project_root, args.run_name)
    report = verify_sprint1(layout)
    if not args.no_write:
        destination = args.output or layout.verification_report
        write_verification_report(report, destination)
        print(f"Sprint 1 verification passed; wrote {destination.resolve()}")
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Sprint1VerificationError as exc:
        print(f"Sprint 1 verification failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


__all__ = [
    "ArtifactLayout",
    "Sprint1VerificationError",
    "layout_for",
    "sha256_file",
    "verify_sprint1",
    "write_verification_report",
]
