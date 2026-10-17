#!/usr/bin/env python3
"""Train, evaluate, and package the Sprint 2 AML model bundle.

The publication path always starts from the complete governed transaction CSV.
Only training negatives are sampled; chronological validation and test periods
are scored in full.  ``--quick`` uses an in-memory deterministic fixture for a
fast dependency and artifact smoke test.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.ml.artifacts import (  # noqa: E402
    save_model_bundle,
    verify_model_bundle,
)
from backend.app.ml.evaluation import (  # noqa: E402
    choose_threshold,
    evaluate_predictions,
)
from backend.app.ml.training import (  # noqa: E402
    SplitScores,
    TrainingConfig,
    TrainingResult,
    train_from_raw,
)


DEFAULT_CONFIG = PROJECT_ROOT / "config" / "model.json"


def load_configuration(path: str | Path) -> dict[str, Any]:
    """Load a strict JSON object used as the reproducible training contract."""
    source = Path(path).resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"model configuration is unreadable: {source}") from error
    if not isinstance(value, dict):
        raise ValueError("model configuration must be a JSON object")
    if int(value.get("schema_version", -1)) != 1:
        raise ValueError("unsupported model configuration schema_version")
    return value


def make_toy_source(rows: int = 420) -> pd.DataFrame:
    """Build a learnable, chronological AMLworld-shaped smoke fixture."""
    if rows < 120:
        raise ValueError("toy source needs at least 120 rows")
    position = np.arange(rows)
    labels = ((position % 17 == 0) | (position % 43 == 0)).astype(np.uint8)
    # Guarantee positives near both chronological edges even if rows changes.
    labels[[1, rows // 2, rows - 2]] = 1
    amount = np.where(
        labels == 1,
        75_000.0 + (position % 23) * 500.0,
        25.0 + (position % 101) * 3.5,
    )
    timestamps = pd.date_range("2025-01-01", periods=rows, freq="min", tz="UTC")
    source_accounts = [f"8{index % 37:08X}" for index in position]
    destination_accounts = [
        source_accounts[index] if index % 19 == 0 else f"9{index % 53:08X}"
        for index in position
    ]
    return pd.DataFrame(
        {
            "Timestamp": timestamps.strftime("%Y/%m/%d %H:%M"),
            "From Bank": [f"{index % 11:03d}" for index in position],
            "Account": source_accounts,
            "To Bank": [f"{(index + labels[index] * 7) % 13:03d}" for index in position],
            "Account.1": destination_accounts,
            "Amount Received": [f"{value:.2f}" for value in amount],
            "Receiving Currency": np.where(labels == 1, "Euro", "US Dollar"),
            "Amount Paid": [f"{value:.2f}" for value in amount],
            "Payment Currency": np.where(labels == 1, "Yuan", "US Dollar"),
            "Payment Format": np.where(labels == 1, "Cash", "ACH"),
            "Is Laundering": labels.astype(str),
        }
    )


def reject_development_source(path: str | Path) -> Path:
    """Require the governed raw CSV and explain the dev-100000 rejection."""
    source = Path(path).resolve()
    lowered_parts = {part.casefold() for part in source.parts}
    if source.is_dir() or "dev-100000" in lowered_parts:
        run_root = source
        if source.is_file():
            parts = list(source.parents)
            run_root = next(
                (parent for parent in parts if parent.name.casefold() == "dev-100000"),
                source.parent,
            )
        test_path = run_root / "model_matrix" / "test.parquet"
        if test_path.is_file():
            test_labels = pd.read_parquet(test_path, columns=["label"])["label"]
            positives = int(pd.to_numeric(test_labels, errors="coerce").fillna(0).sum())
            if positives == 0:
                raise ValueError(
                    "refusing dev-100000: its chronological test split has zero "
                    "positive labels, so model evaluation would be invalid; train "
                    "from data/raw/ibm_aml_hi_small/HI-Small_Trans.csv"
                )
        raise ValueError(
            "Sprint 2 publication training requires the complete governed raw "
            "transaction CSV, not a bounded development run"
        )
    if source.suffix.casefold() != ".csv":
        raise ValueError(
            "Sprint 2 training input must be the complete governed transaction CSV"
        )
    if not source.is_file():
        raise FileNotFoundError(source)
    return source


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _quick_config(config: TrainingConfig) -> TrainingConfig:
    lightgbm = dict(config.lightgbm_parameters)
    lightgbm.update(
        {
            "num_boost_round": 40,
            "early_stopping_rounds": 8,
            "min_data_in_leaf": 5,
            "num_leaves": 15,
            "num_threads": 1,
            "feature_fraction": 1.0,
            "bagging_fraction": 1.0,
            "bagging_freq": 0,
        }
    )
    isolation = dict(config.isolation_forest_parameters)
    isolation.update(
        {"number_of_estimators": 24, "maximum_samples": 128, "jobs": 1}
    )
    return replace(
        config,
        raw_input=None,
        chunk_size=64,
        minimum_positives={"train": 2, "validation": 2, "test": 2},
        negative_to_positive_ratio=4.0,
        lightgbm_parameters=lightgbm,
        isolation_forest_parameters=isolation,
    )


def _pattern_case_context(
    path: Path | None,
    result: TrainingResult,
) -> dict[str, Any] | None:
    """Build evaluation-only fingerprint/case maps from governed memberships."""
    if path is None or not path.is_file():
        return None
    memberships = pd.read_parquet(
        path,
        columns=["case_id", "transaction_fingerprint", "Timestamp"],
    )
    required = {"case_id", "transaction_fingerprint", "Timestamp"}
    if set(memberships.columns) != required:
        raise ValueError("pattern membership artifact has an incompatible schema")
    memberships = memberships.copy()
    memberships["split"] = result.boundaries.assign(memberships["Timestamp"])
    lookups: dict[str, dict[str, tuple[str, ...]]] = {}
    for split in ("validation", "test"):
        selected = memberships.loc[memberships["split"] == split]
        lookups[split] = {
            str(fingerprint): tuple(sorted(set(group["case_id"].astype(str))))
            for fingerprint, group in selected.groupby(
                "transaction_fingerprint", sort=False
            )
        }
    train_cases = set(
        memberships.loc[memberships["split"] == "train", "case_id"].astype(str)
    )
    validation_cases = set(
        memberships.loc[
            memberships["split"] == "validation", "case_id"
        ].astype(str)
    )
    return {
        "lookups": lookups,
        "known": {
            "validation": train_cases,
            "test": train_cases | validation_cases,
        },
        "membership_rows": len(memberships),
    }


def _score_metrics(
    scores: SplitScores,
    *,
    threshold: float,
    capacity_per_hour: int,
    case_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    metadata = scores.metadata
    case_ids = None
    known_case_ids = None
    matched_positive_fingerprints = 0
    if case_context is not None and "transaction_fingerprint" in metadata:
        lookup = case_context["lookups"][scores.split]
        case_ids = [
            lookup.get(str(value), ()) if value is not None and value is not pd.NA else ()
            for value in metadata["transaction_fingerprint"]
        ]
        known_case_ids = case_context["known"][scores.split]
        positive_fingerprints = metadata.loc[
            scores.labels == 1, "transaction_fingerprint"
        ].dropna()
        matched_positive_fingerprints = int(
            positive_fingerprints.astype(str).isin(lookup).sum()
        )
    metrics = evaluate_predictions(
        scores.labels,
        scores.probabilities,
        threshold=threshold,
        event_times=metadata["event_time"],
        event_ids=metadata["event_id"],
        capacity_per_hour=capacity_per_hour,
        case_ids=case_ids,
        known_case_ids=known_case_ids,
        amounts=(
            metadata["transaction_amount"]
            if "transaction_amount" in metadata
            else None
        ),
    )
    metrics["full_split_scored"] = True
    if case_context is not None:
        metrics["case_mapping"] = {
            "positive_rows": int(scores.labels.sum()),
            "matched_positive_fingerprints": matched_positive_fingerprints,
        }
    return metrics


def train_and_package(
    configuration: Mapping[str, Any],
    *,
    source_override: str | Path | None = None,
    output_override: str | Path | None = None,
    quick: bool = False,
    enable_xgboost: bool = False,
    bundle_id: str | None = None,
) -> dict[str, Any]:
    """Execute the complete deterministic training and packaging workflow."""
    options = TrainingConfig.from_mapping(configuration)
    if quick:
        source: Path | pd.DataFrame = make_toy_source()
        options = _quick_config(options)
    else:
        configured_source = source_override or options.raw_input
        if configured_source is None:
            raise ValueError("configuration does not define raw.input")
        source = reject_development_source(_project_path(configured_source))
        options = replace(options, raw_input=source)
    if enable_xgboost:
        options = replace(options, xgboost_crosscheck_enabled=True)

    output_section = configuration.get("output", {})
    if not isinstance(output_section, Mapping):
        raise TypeError("output must be a mapping")
    configured_output = output_section.get(
        "directory", "artifacts/models/sprint2-lightgbm"
    )
    output = _project_path(output_override or configured_output)
    if quick and output_override is None:
        output = output.with_name(output.name + "-smoke")

    result = train_from_raw(source, options)
    threshold_selection = choose_threshold(
        result.validation.labels,
        result.validation.probabilities,
        beta=1.0,
    )
    threshold = threshold_selection["threshold"]

    membership_path: Path | None = None
    raw_section = configuration.get("raw", {})
    if not quick and isinstance(raw_section, Mapping):
        membership_value = raw_section.get("pattern_memberships")
        if membership_value:
            membership_path = _project_path(str(membership_value))
    case_context = _pattern_case_context(membership_path, result)
    metrics = {
        "schema_version": 1,
        "primary_model": "lightgbm",
        "model_selection": {
            "selected": "lightgbm",
            "basis": (
                "CPU-efficient gradient boosting for imbalanced tabular data, "
                "with native TreeSHAP support and deterministic training"
            ),
            "logistic_baseline_trained": "logistic_baseline"
            in result.supplementary_models,
            "isolation_forest_trained": "isolation_forest"
            in result.supplementary_models,
            "xgboost_crosscheck_trained": "xgboost_crosscheck"
            in result.supplementary_models,
        },
        "threshold_selection": {
            "split": "validation",
            "metric": "f1",
            **threshold_selection,
        },
        "validation": _score_metrics(
            result.validation,
            threshold=threshold,
            capacity_per_hour=options.capacity_per_hour,
            case_context=case_context,
        ),
        "test": _score_metrics(
            result.test,
            threshold=threshold,
            capacity_per_hour=options.capacity_per_hour,
            case_context=case_context,
        ),
    }
    if case_context is not None:
        metrics["case_evaluation"] = {
            "source": str(membership_path),
            "membership_rows": case_context["membership_rows"],
            "ground_truth_used_after_scoring_only": True,
        }

    training_record = result.manifest()
    training_record["best_iteration"] = result.model.best_iteration
    training_record["quick_smoke_mode"] = quick
    training_record["source"] = "in-memory-toy" if quick else str(source)
    effective_configuration = {
        "schema_version": 1,
        "source_configuration": dict(configuration),
        "effective": {
            "chunk_size": options.chunk_size,
            "split_ratios": dict(options.split_ratios),
            "minimum_positives": dict(options.minimum_positives),
            "negative_to_positive_ratio": options.negative_to_positive_ratio,
            "seed": options.seed,
            "feature_columns": list(options.feature_columns),
            "categorical_features": list(options.categorical_features),
            "inverse_probability_weighting": options.inverse_probability_weighting,
            "xgboost_crosscheck_enabled": options.xgboost_crosscheck_enabled,
            "quick_smoke_mode": quick,
        },
    }
    supplementary = result.supplementary_models
    bundle = save_model_bundle(
        output,
        result.model,
        feature_dtypes={name: "float32" for name in result.feature_columns},
        metrics=metrics,
        training=training_record,
        configuration=effective_configuration,
        calibrator=result.calibrator,
        logistic_baseline=supplementary.get("logistic_baseline"),
        anomaly_signal=supplementary.get("isolation_forest"),
        xgboost_crosscheck=supplementary.get("xgboost_crosscheck"),
        bundle_id=bundle_id,
    )
    verified = verify_model_bundle(bundle)
    return {
        "status": "complete",
        "bundle": str(bundle),
        "bundle_id": verified.manifest["bundle_id"],
        "files": sorted(verified.manifest["files"]),
        "quick_smoke_mode": quick,
        "best_iteration": result.model.best_iteration,
        "threshold": threshold,
        "validation": {
            key: metrics["validation"][key]
            for key in ("rows", "positives", "pr_auc", "roc_auc", "f1")
        },
        "test": {
            key: metrics["test"][key]
            for key in ("rows", "positives", "pr_auc", "roc_auc", "f1")
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--source",
        help="override raw.input; must be the complete governed transaction CSV",
    )
    parser.add_argument("--output", help="new model bundle directory")
    parser.add_argument("--bundle-id", help="stable deployment identifier")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="train on a deterministic toy source for a fast end-to-end smoke test",
    )
    parser.add_argument(
        "--xgboost",
        action="store_true",
        help="also train and persist the optional native XGBoost cross-check",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        configuration = load_configuration(_project_path(arguments.config))
        summary = train_and_package(
            configuration,
            source_override=arguments.source,
            output_override=arguments.output,
            quick=arguments.quick,
            enable_xgboost=arguments.xgboost,
            bundle_id=arguments.bundle_id,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"Sprint 2 training failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
