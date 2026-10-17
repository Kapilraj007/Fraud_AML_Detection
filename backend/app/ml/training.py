"""Resource-aware orchestration for the AMLworld baseline model.

The core supervised model is LightGBM.  Training retains every positive and a
deterministic, exact number of negative examples, while chronological
validation and test rows remain unsampled.  Model-layer imports are delayed
until fitting so the data/split helpers remain usable in small tests without
LightGBM or scikit-learn import side effects.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import importlib
import inspect
import math
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from backend.app.ml.raw_features import (
    DEFAULT_SPLIT_RATIOS,
    RAW_FEATURE_COLUMNS,
    RAW_FEATURE_SET_VERSION,
    RAW_METADATA_COLUMNS,
    RawFeatureChunk,
    RawSource,
    SplitBoundaries,
    deterministic_sample_keys,
    iter_raw_feature_chunks,
    plan_chronological_splits,
)


DEFAULT_LIGHTGBM_PARAMETERS: dict[str, Any] = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.90,
    "bagging_fraction": 0.80,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "num_boost_round": 500,
    "early_stopping_rounds": 50,
    "deterministic": True,
    "force_col_wise": True,
    "verbosity": -1,
}


class ScoreModel(Protocol):
    """Minimal prediction protocol accepted by streaming evaluation."""

    def predict_probability(self, features: pd.DataFrame) -> Any: ...

    def predict_margin(self, features: pd.DataFrame) -> Any: ...


@dataclass(frozen=True)
class TrainingConfig:
    """Validated configuration for raw chronological model training."""

    raw_input: Path | None = None
    chunk_size: int = 100_000
    split_ratios: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_SPLIT_RATIOS)
    )
    negative_to_positive_ratio: float = 20.0
    seed: int = 20261007
    minimum_positives: int | Mapping[str, int] = 1
    feature_columns: tuple[str, ...] = RAW_FEATURE_COLUMNS
    categorical_features: tuple[str, ...] = ()
    lightgbm_parameters: Mapping[str, Any] = field(
        default_factory=lambda: dict(DEFAULT_LIGHTGBM_PARAMETERS)
    )
    calibration_enabled: bool = True
    logistic_baseline_enabled: bool = False
    logistic_baseline_parameters: Mapping[str, Any] = field(default_factory=dict)
    isolation_forest_enabled: bool = False
    isolation_forest_parameters: Mapping[str, Any] = field(default_factory=dict)
    xgboost_crosscheck_enabled: bool = False
    xgboost_parameters: Mapping[str, Any] = field(default_factory=dict)
    inverse_probability_weighting: bool = True
    capacity_per_hour: int = 25

    def __post_init__(self) -> None:
        if self.raw_input is not None:
            object.__setattr__(self, "raw_input", Path(self.raw_input))
        chunk_size = int(self.chunk_size)
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least one")
        object.__setattr__(self, "chunk_size", chunk_size)

        ratios = _validated_ratios(self.split_ratios)
        object.__setattr__(self, "split_ratios", ratios)
        ratio = float(self.negative_to_positive_ratio)
        if not math.isfinite(ratio) or ratio <= 0:
            raise ValueError(
                "negative_to_positive_ratio must be positive and finite"
            )
        object.__setattr__(self, "negative_to_positive_ratio", ratio)
        object.__setattr__(self, "seed", int(self.seed))
        minimum = _minimum_positive_counts(self.minimum_positives)
        object.__setattr__(self, "minimum_positives", minimum)

        features = _validate_feature_columns(self.feature_columns)
        object.__setattr__(self, "feature_columns", features)
        categorical = tuple(str(name) for name in self.categorical_features)
        if len(categorical) != len(set(categorical)):
            raise ValueError("categorical_features contains duplicates")
        unknown_categorical = set(categorical) - set(features)
        if unknown_categorical:
            raise ValueError(
                "categorical_features are not model features: "
                f"{sorted(unknown_categorical)}"
            )
        object.__setattr__(self, "categorical_features", categorical)
        if not isinstance(self.lightgbm_parameters, Mapping):
            raise TypeError("lightgbm_parameters must be a mapping")
        parameters = {**DEFAULT_LIGHTGBM_PARAMETERS, **dict(self.lightgbm_parameters)}
        parameters.setdefault("seed", self.seed)
        parameters.setdefault("feature_fraction_seed", self.seed)
        parameters.setdefault("bagging_seed", self.seed)
        parameters.setdefault("data_random_seed", self.seed)
        if bool(parameters.get("force_row_wise")):
            parameters["force_col_wise"] = False
        elif bool(parameters.get("force_col_wise")):
            parameters["force_row_wise"] = False
        if categorical:
            parameters["categorical_feature"] = list(categorical)
        object.__setattr__(self, "lightgbm_parameters", parameters)
        for name in (
            "logistic_baseline_parameters",
            "isolation_forest_parameters",
            "xgboost_parameters",
        ):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping")
            object.__setattr__(self, name, dict(value))
        object.__setattr__(
            self,
            "inverse_probability_weighting",
            bool(self.inverse_probability_weighting),
        )
        capacity = int(self.capacity_per_hour)
        if capacity < 1:
            raise ValueError("capacity_per_hour must be at least one")
        object.__setattr__(self, "capacity_per_hour", capacity)

    def minimum_for(self, split: str) -> int:
        """Return the configured minimum positive count for one split."""
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        return int(self.minimum_positives[split])  # type: ignore[index]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "TrainingConfig":
        """Read both the Sprint 2 nested JSON shape and flat test fixtures."""
        if not isinstance(raw, Mapping):
            raise TypeError("training config must be a mapping")
        root = dict(raw)
        raw_section = _mapping(root.get("raw"), "raw")
        split_section = _mapping(root.get("split"), "split")
        sampling = _mapping(root.get("sampling"), "sampling")
        evaluation = _mapping(root.get("evaluation"), "evaluation")

        input_value = raw_section.get("input", root.get("raw_input"))
        chunk_size = raw_section.get("chunk_size", root.get("chunk_size", 100_000))
        ratios = (
            split_section.get("ratios")
            or root.get("split_ratios")
            or root.get("ratios")
            or DEFAULT_SPLIT_RATIOS
        )
        sampling_ratio = sampling.get(
            "negative_to_positive_ratio",
            root.get("negative_to_positive_ratio", 20.0),
        )
        seed = sampling.get("seed", root.get("seed", 20261007))
        minimum = split_section.get(
            "minimum_positives",
            root.get("minimum_positives", 1),
        )
        features = root.get("feature_columns", RAW_FEATURE_COLUMNS)
        categorical = root.get("categorical_features", ())

        lightgbm = root.get("lightgbm", {})
        if lightgbm is None:
            lightgbm = {}
        if not isinstance(lightgbm, Mapping):
            raise TypeError("lightgbm must be a mapping")
        parameters = lightgbm.get("parameters", lightgbm)
        if not isinstance(parameters, Mapping):
            raise TypeError("lightgbm parameters must be a mapping")
        parameters = {
            key: value for key, value in parameters.items()
            if key not in {"enabled", "parameters"}
        }

        calibration = root.get("calibration", True)
        calibration_enabled = _enabled(calibration, default=True)
        logistic_section = root.get("logistic_baseline", False)
        isolation_section = root.get("isolation_forest", False)
        xgboost_section = root.get("xgboost_crosscheck", False)
        logistic_enabled = _enabled(logistic_section, default=False)
        isolation_enabled = _enabled(isolation_section, default=False)
        xgboost_enabled = _enabled(xgboost_section, default=False)
        logistic_parameters = _component_parameters(logistic_section)
        isolation_parameters = _component_parameters(isolation_section)
        xgboost_parameters = _component_parameters(xgboost_section)
        inverse_weighting = sampling.get(
            "inverse_probability_weighting",
            root.get("inverse_probability_weighting", True),
        )
        capacity = evaluation.get(
            "capacity_per_hour", root.get("capacity_per_hour", 25)
        )
        return cls(
            raw_input=None if input_value is None else Path(input_value),
            chunk_size=chunk_size,
            split_ratios=ratios,
            negative_to_positive_ratio=sampling_ratio,
            seed=seed,
            minimum_positives=minimum,
            feature_columns=tuple(features),
            categorical_features=tuple(categorical),
            lightgbm_parameters=parameters,
            calibration_enabled=calibration_enabled,
            logistic_baseline_enabled=logistic_enabled,
            logistic_baseline_parameters=logistic_parameters,
            isolation_forest_enabled=isolation_enabled,
            isolation_forest_parameters=isolation_parameters,
            xgboost_crosscheck_enabled=xgboost_enabled,
            xgboost_parameters=xgboost_parameters,
            inverse_probability_weighting=inverse_weighting,
            capacity_per_hour=capacity,
        )


@dataclass
class ModelBatch:
    """An in-memory model batch with aligned metadata and importance weights."""

    features: pd.DataFrame
    labels: np.ndarray
    metadata: pd.DataFrame
    sample_weight: np.ndarray | None = None

    def __post_init__(self) -> None:
        rows = len(self.features)
        if len(self.labels) != rows or len(self.metadata) != rows:
            raise ValueError("model batch components are not row-aligned")
        if self.sample_weight is not None and len(self.sample_weight) != rows:
            raise ValueError("sample weights are not row-aligned")


@dataclass
class SplitScores:
    """Complete labels/scores with compact row identity for one holdout."""

    split: str
    probabilities: np.ndarray
    labels: np.ndarray
    metadata: pd.DataFrame
    raw_margins: np.ndarray | None = None

    def __post_init__(self) -> None:
        rows = len(self.probabilities)
        if len(self.labels) != rows or len(self.metadata) != rows:
            raise ValueError("split score components are not row-aligned")
        if self.raw_margins is not None and len(self.raw_margins) != rows:
            raise ValueError("raw margins are not row-aligned")


@dataclass
class TrainingResult:
    """Fitted model state plus full chronological validation/test scores."""

    model: Any
    calibrator: Any | None
    boundaries: SplitBoundaries
    feature_columns: tuple[str, ...]
    validation: SplitScores
    test: SplitScores
    training_summary: dict[str, Any]
    supplementary_models: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        """Return the serializable portion expected by an artifact writer."""
        return {
            "model_family": "lightgbm",
            "feature_set": RAW_FEATURE_SET_VERSION,
            "feature_columns": list(self.feature_columns),
            "calibrated": self.calibrator is not None,
            "split_plan": self.boundaries.to_dict(),
            "training": dict(self.training_summary),
            "validation_rows_scored": len(self.validation.labels),
            "test_rows_scored": len(self.test.labels),
            "supplementary_models": sorted(self.supplementary_models),
        }


def validate_binary_split(
    labels: Sequence[Any] | np.ndarray | pd.Series,
    split: str,
    *,
    minimum_positives: int = 1,
    minimum_negatives: int = 1,
) -> dict[str, int]:
    """Fail fast for null, non-binary, or single-class model splits."""
    array = np.asarray(labels)
    if array.ndim != 1:
        raise ValueError(f"{split} labels must be one-dimensional")
    if len(array) == 0:
        raise ValueError(f"{split} split is empty")
    numeric = pd.to_numeric(pd.Series(array), errors="coerce")
    if numeric.isna().any() or not numeric.isin([0, 1]).all():
        raise ValueError(f"{split} labels must contain only binary 0/1 values")
    positives = int(numeric.sum())
    negatives = len(numeric) - positives
    minimum_positives = int(minimum_positives)
    minimum_negatives = int(minimum_negatives)
    if minimum_positives < 1 or minimum_negatives < 1:
        raise ValueError("minimum class counts must be at least one")
    if positives < minimum_positives:
        raise ValueError(
            f"{split} has {positives} positives; at least "
            f"{minimum_positives} are required"
        )
    if negatives < minimum_negatives:
        raise ValueError(
            f"{split} has {negatives} negatives; at least "
            f"{minimum_negatives} are required"
        )
    return {"rows": len(numeric), "positives": positives, "negatives": negatives}


def collect_training_sample(
    source: RawSource,
    boundaries: SplitBoundaries,
    *,
    negative_to_positive_ratio: float = 20.0,
    seed: int = 20261007,
    chunk_size: int = 100_000,
    feature_columns: Sequence[str] | None = None,
    minimum_positives: int = 1,
) -> ModelBatch:
    """Collect all train positives and exactly ``ratio * positives`` negatives.

    Candidate negatives are reduced to their globally lowest deterministic
    hash keys after every chunk, so memory is bounded by the final training
    sample rather than the 3.5M-row full train period.
    """
    ratio = float(negative_to_positive_ratio)
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("negative_to_positive_ratio must be positive and finite")
    selected_features = _validate_feature_columns(
        RAW_FEATURE_COLUMNS if feature_columns is None else feature_columns
    )
    full_positive_count = boundaries.positive_count("train")
    target_negatives = min(
        boundaries.negative_count("train"),
        int(math.ceil(full_positive_count * ratio)),
    )
    if target_negatives < 1:
        raise ValueError("training split does not contain sampleable negatives")

    positives: list[pd.DataFrame] = []
    negative_candidates: pd.DataFrame | None = None
    observed_rows = 0
    observed_positives = 0
    for chunk in iter_raw_feature_chunks(
        source,
        boundaries,
        split="train",
        chunk_size=chunk_size,
        feature_columns=selected_features,
    ):
        combined = _combined_frame(chunk)
        observed_rows += len(combined)
        positive_mask = combined["label"].to_numpy(dtype=np.uint8) == 1
        observed_positives += int(positive_mask.sum())
        if bool(positive_mask.any()):
            positives.append(combined.loc[positive_mask].copy())
        negatives = combined.loc[~positive_mask].copy()
        if not negatives.empty:
            negatives["__sample_key"] = deterministic_sample_keys(
                negatives["source_row_number"], seed=seed
            )
            if len(negatives) > target_negatives:
                negatives = negatives.nsmallest(
                    target_negatives,
                    ["__sample_key", "source_row_number"],
                    keep="first",
                )
            negative_candidates = (
                negatives
                if negative_candidates is None
                else pd.concat(
                    [negative_candidates, negatives], ignore_index=True
                ).nsmallest(
                    target_negatives,
                    ["__sample_key", "source_row_number"],
                    keep="first",
                )
            )

    if observed_rows != boundaries.count("train"):
        raise RuntimeError(
            f"train source changed after split planning: expected "
            f"{boundaries.count('train')} rows, observed {observed_rows}"
        )
    if observed_positives != full_positive_count:
        raise RuntimeError(
            "train positive count changed after chronological split planning"
        )
    if not positives or negative_candidates is None:
        raise ValueError("training split must contain both binary classes")
    positive_frame = pd.concat(positives, ignore_index=True)
    negative_frame = negative_candidates.drop(columns="__sample_key")
    if len(negative_frame) != target_negatives:
        raise RuntimeError("could not collect the requested training negatives")
    sample = pd.concat([positive_frame, negative_frame], ignore_index=True)
    sample = sample.sort_values(
        ["event_time", "source_row_number"], kind="mergesort"
    ).reset_index(drop=True)
    labels = sample["label"].to_numpy(dtype=np.uint8)
    counts = validate_binary_split(
        labels,
        "sampled train",
        minimum_positives=minimum_positives,
        minimum_negatives=1,
    )

    # Inverse-probability weights restore the full chronological train prior.
    positive_weight = full_positive_count / counts["positives"]
    negative_weight = boundaries.negative_count("train") / counts["negatives"]
    weights = np.where(labels == 1, positive_weight, negative_weight).astype(
        np.float64
    )
    weights /= float(weights.mean())
    return ModelBatch(
        features=sample.loc[:, list(selected_features)].astype("float32"),
        labels=labels,
        metadata=sample.loc[:, list(RAW_METADATA_COLUMNS)].copy(),
        sample_weight=weights,
    )


def collect_feature_split(
    source: RawSource,
    boundaries: SplitBoundaries,
    split: str,
    *,
    chunk_size: int = 100_000,
    feature_columns: Sequence[str] | None = None,
    minimum_positives: int = 1,
) -> ModelBatch:
    """Collect a complete, unsampled chronological split for model fitting."""
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    selected_features = _validate_feature_columns(
        RAW_FEATURE_COLUMNS if feature_columns is None else feature_columns
    )
    features: list[pd.DataFrame] = []
    labels: list[np.ndarray] = []
    metadata: list[pd.DataFrame] = []
    for chunk in iter_raw_feature_chunks(
        source,
        boundaries,
        split=split,
        chunk_size=chunk_size,
        feature_columns=selected_features,
    ):
        features.append(chunk.features)
        labels.append(chunk.labels)
        metadata.append(chunk.metadata)
    if not features:
        raise ValueError(f"{split} split is empty")
    label_array = np.concatenate(labels).astype(np.uint8, copy=False)
    if len(label_array) != boundaries.count(split):
        raise RuntimeError(
            f"{split} source changed after split planning: expected "
            f"{boundaries.count(split)} rows, observed {len(label_array)}"
        )
    validate_binary_split(
        label_array,
        split,
        minimum_positives=minimum_positives,
        minimum_negatives=1,
    )
    return ModelBatch(
        features=pd.concat(features, ignore_index=True).loc[:, list(selected_features)],
        labels=label_array,
        metadata=pd.concat(metadata, ignore_index=True),
    )


def evaluate_stream(
    model: Any,
    source: RawSource,
    boundaries: SplitBoundaries,
    split: str,
    *,
    calibrator: Any | None = None,
    chunk_size: int = 100_000,
    feature_columns: Sequence[str] | None = None,
    minimum_positives: int = 1,
    sink: Callable[[pd.DataFrame], None] | None = None,
) -> SplitScores:
    """Score every validation/test row while retaining only compact outputs."""
    if split not in {"validation", "test"}:
        raise ValueError("stream evaluation split must be validation or test")
    selected_features = _validate_feature_columns(
        RAW_FEATURE_COLUMNS if feature_columns is None else feature_columns
    )
    probabilities: list[np.ndarray] = []
    margins: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    metadata: list[pd.DataFrame] = []
    for chunk in iter_raw_feature_chunks(
        source,
        boundaries,
        split=split,
        chunk_size=chunk_size,
        feature_columns=selected_features,
    ):
        raw_margin = _predict_margin(model, chunk.features)
        probability = (
            _calibrator_probability(calibrator, raw_margin)
            if calibrator is not None
            else _predict_probability(model, chunk.features, margin=raw_margin)
        )
        _validate_scores(probability, len(chunk.labels), split)
        compact = chunk.metadata.loc[:, list(RAW_METADATA_COLUMNS)].copy()
        if sink is not None:
            payload = compact.copy()
            payload["raw_margin"] = raw_margin
            payload["probability"] = probability
            sink(payload)
        probabilities.append(probability)
        margins.append(raw_margin)
        labels.append(chunk.labels)
        metadata.append(compact)

    if not probabilities:
        raise ValueError(f"{split} split is empty")
    probability_array = np.concatenate(probabilities)
    margin_array = np.concatenate(margins)
    label_array = np.concatenate(labels).astype(np.uint8, copy=False)
    metadata_frame = pd.concat(metadata, ignore_index=True)
    if len(label_array) != boundaries.count(split):
        raise RuntimeError(
            f"{split} source changed after split planning: expected "
            f"{boundaries.count(split)} rows, observed {len(label_array)}"
        )
    validate_binary_split(
        label_array,
        split,
        minimum_positives=minimum_positives,
        minimum_negatives=1,
    )
    return SplitScores(
        split=split,
        probabilities=probability_array,
        labels=label_array,
        metadata=metadata_frame,
        raw_margins=margin_array,
    )


def train_from_raw(
    source: RawSource | None = None,
    config: TrainingConfig | Mapping[str, Any] | None = None,
    *,
    trainer: Callable[..., Any] | None = None,
    model_layer: Any | None = None,
) -> TrainingResult:
    """Fit, calibrate, and score the LightGBM raw-feature baseline.

    ``trainer`` is an optional dependency-injection hook for toy tests.  Its
    production contract is ``(train_df, y, val_df, y_val, *, parameters,
    sample_weight)``.  When omitted, ``backend.app.ml.model.train_lightgbm`` is
    imported lazily.
    """
    options = _training_config(config)
    actual_source = source if source is not None else options.raw_input
    if actual_source is None:
        raise ValueError("a raw AMLworld source path or DataFrame is required")
    if not _is_repeatable_source(actual_source):
        raise TypeError(
            "train_from_raw requires a repeatable CSV path or DataFrame; "
            "one-shot iterables can be used with lower-level helpers"
        )

    boundaries = plan_chronological_splits(
        actual_source,
        chunk_size=options.chunk_size,
        ratios=options.split_ratios,
        minimum_positives=options.minimum_positives,
        require_both_classes=True,
    )
    train_batch = collect_training_sample(
        actual_source,
        boundaries,
        negative_to_positive_ratio=options.negative_to_positive_ratio,
        seed=options.seed,
        chunk_size=options.chunk_size,
        feature_columns=options.feature_columns,
        minimum_positives=options.minimum_for("train"),
    )
    # The full validation matrix is modest for HI-Small (~0.76M x 23 float32)
    # and is required by LightGBM for temporal early stopping.  Test remains
    # chunk-scored and neither holdout is undersampled.
    validation_batch = collect_feature_split(
        actual_source,
        boundaries,
        "validation",
        chunk_size=options.chunk_size,
        feature_columns=options.feature_columns,
        minimum_positives=options.minimum_for("validation"),
    )

    layer = model_layer or _model_layer()
    fit = trainer or getattr(layer, "train_lightgbm")
    model = _fit_lightgbm(fit, train_batch, validation_batch, options)
    raw_validation_margin = _predict_margin(model, validation_batch.features)
    calibrator = None
    if options.calibration_enabled:
        calibrator_type = getattr(layer, "PlattCalibrator")
        calibrator = calibrator_type.fit(
            raw_validation_margin, validation_batch.labels
        )
        if hasattr(model, "calibrator"):
            model.calibrator = calibrator
    validation_probability = (
        _calibrator_probability(calibrator, raw_validation_margin)
        if calibrator is not None
        else _predict_probability(
            model, validation_batch.features, margin=raw_validation_margin
        )
    )
    _validate_scores(
        validation_probability, len(validation_batch.labels), "validation"
    )
    validation_scores = SplitScores(
        split="validation",
        probabilities=validation_probability,
        labels=validation_batch.labels,
        metadata=validation_batch.metadata,
        raw_margins=raw_validation_margin,
    )
    test_scores = evaluate_stream(
        model,
        actual_source,
        boundaries,
        "test",
        calibrator=calibrator,
        chunk_size=options.chunk_size,
        feature_columns=options.feature_columns,
        minimum_positives=options.minimum_for("test"),
    )

    supplementary: dict[str, Any] = {}
    if options.logistic_baseline_enabled:
        helper = getattr(layer, "train_logistic_baseline")
        logistic_parameters = dict(options.logistic_baseline_parameters)
        logistic_parameters.setdefault("seed", options.seed)
        supplementary["logistic_baseline"] = _invoke_optional_trainer(
            helper,
            train_batch.features,
            train_batch.labels,
            validation_batch.features,
            validation_batch.labels,
            sample_weight=(
                train_batch.sample_weight
                if options.inverse_probability_weighting
                else None
            ),
            **logistic_parameters,
        )
    if options.isolation_forest_enabled:
        helper = getattr(layer, "train_isolation_forest")
        isolation_parameters = dict(options.isolation_forest_parameters)
        isolation_parameters.setdefault("seed", options.seed)
        supplementary["isolation_forest"] = _invoke_optional_trainer(
            helper,
            train_batch.features,
            **isolation_parameters,
        )
    if options.xgboost_crosscheck_enabled:
        helper = getattr(layer, "train_xgboost_crosscheck")
        supplementary["xgboost_crosscheck"] = _invoke_optional_trainer(
            helper,
            train_batch.features,
            train_batch.labels,
            validation_batch.features,
            validation_batch.labels,
            parameters=dict(options.xgboost_parameters),
            sample_weight=(
                train_batch.sample_weight
                if options.inverse_probability_weighting
                else None
            ),
        )

    sampled_counts = validate_binary_split(
        train_batch.labels,
        "sampled train",
        minimum_positives=options.minimum_for("train"),
        minimum_negatives=1,
    )
    summary = {
        "algorithm": "lightgbm",
        "feature_set": RAW_FEATURE_SET_VERSION,
        "seed": options.seed,
        "full_train_rows": boundaries.count("train"),
        "full_train_positives": boundaries.positive_count("train"),
        "full_train_negatives": boundaries.negative_count("train"),
        "sampled_train_rows": sampled_counts["rows"],
        "sampled_train_positives": sampled_counts["positives"],
        "sampled_train_negatives": sampled_counts["negatives"],
        "negative_to_positive_ratio": options.negative_to_positive_ratio,
        "inverse_probability_weighting": options.inverse_probability_weighting,
        "categorical_features": list(options.categorical_features),
        "validation_rows": len(validation_batch.labels),
        "test_rows": len(test_scores.labels),
        "train_only_negative_undersampling": True,
    }
    return TrainingResult(
        model=model,
        calibrator=calibrator,
        boundaries=boundaries,
        feature_columns=options.feature_columns,
        validation=validation_scores,
        test=test_scores,
        training_summary=summary,
        supplementary_models=supplementary,
    )


def _fit_lightgbm(
    trainer: Callable[..., Any],
    train: ModelBatch,
    validation: ModelBatch,
    config: TrainingConfig,
) -> Any:
    fitted = trainer(
        train.features,
        train.labels,
        validation.features,
        validation.labels,
        parameters=dict(config.lightgbm_parameters),
        sample_weight=(
            train.sample_weight
            if config.inverse_probability_weighting
            else None
        ),
    )
    if isinstance(fitted, tuple):
        if not fitted:
            raise RuntimeError("LightGBM trainer returned an empty tuple")
        return fitted[0]
    if fitted is None:
        raise RuntimeError("LightGBM trainer returned no model")
    return fitted


def _invoke_optional_trainer(helper: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call optional model helpers without imposing unused keyword names."""
    signature = inspect.signature(helper)
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    accepted = (
        kwargs
        if accepts_kwargs
        else {key: value for key, value in kwargs.items()
              if key in signature.parameters}
    )
    positional_capacity = sum(
        parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        for parameter in signature.parameters.values()
    )
    return helper(*args[:positional_capacity], **accepted)


def _predict_margin(model: Any, features: pd.DataFrame) -> np.ndarray:
    for name in ("predict_margin", "decision_function"):
        method = getattr(model, name, None)
        if callable(method):
            return _one_dimensional_scores(method(features), name)
    probability = _predict_probability(model, features)
    clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
    return np.log(clipped / (1.0 - clipped))


def _predict_probability(
    model: Any,
    features: pd.DataFrame,
    *,
    margin: np.ndarray | None = None,
) -> np.ndarray:
    for name in ("predict_probability", "predict_proba", "predict_scores"):
        method = getattr(model, name, None)
        if callable(method):
            values = np.asarray(method(features), dtype=np.float64)
            if values.ndim == 2 and values.shape[1] == 2:
                values = values[:, 1]
            return _one_dimensional_scores(values, name)
    if margin is not None:
        return _sigmoid(margin)
    raise TypeError(
        "model must provide predict_probability, predict_proba, "
        "predict_scores, or predict_margin"
    )


def _calibrator_probability(calibrator: Any, margin: np.ndarray) -> np.ndarray:
    for name in ("predict_probability", "predict_proba", "transform", "predict"):
        method = getattr(calibrator, name, None)
        if callable(method):
            values = np.asarray(method(margin), dtype=np.float64)
            if values.ndim == 2 and values.shape[1] == 2:
                values = values[:, 1]
            return _one_dimensional_scores(values, f"calibrator.{name}")
    raise TypeError("calibrator does not expose a probability prediction method")


def _one_dimensional_scores(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1:
        raise ValueError(f"{name} must return one score per row")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} returned non-finite scores")
    return array


def _validate_scores(values: np.ndarray, expected: int, split: str) -> None:
    if len(values) != expected:
        raise ValueError(
            f"{split} scorer returned {len(values)} values for {expected} rows"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{split} probabilities contain non-finite values")
    if bool(((values < 0.0) | (values > 1.0)).any()):
        raise ValueError(f"{split} probabilities must lie between zero and one")


def _sigmoid(margin: np.ndarray) -> np.ndarray:
    values = np.asarray(margin, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result


def _combined_frame(chunk: RawFeatureChunk) -> pd.DataFrame:
    metadata = chunk.metadata.reset_index(drop=True)
    features = chunk.features.reset_index(drop=True)
    overlap = set(metadata.columns) & set(features.columns)
    if overlap:
        raise ValueError(f"feature/metadata columns overlap: {sorted(overlap)}")
    return pd.concat([metadata, features], axis=1)


def _validate_feature_columns(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("feature_columns must be a sequence")
    result = tuple(map(str, values))
    if not result:
        raise ValueError("feature_columns cannot be empty")
    if len(result) != len(set(result)):
        raise ValueError("feature_columns contains duplicates")
    unknown = set(result) - set(RAW_FEATURE_COLUMNS)
    if unknown:
        raise ValueError(f"unknown raw feature columns: {sorted(unknown)}")
    return result


def _validated_ratios(values: Mapping[str, Any]) -> dict[str, float]:
    if not isinstance(values, Mapping) or set(values) != {
        "train", "validation", "test"
    }:
        raise ValueError("split_ratios must contain train, validation, and test")
    result = {name: float(values[name]) for name in (
        "train", "validation", "test"
    )}
    if any(not math.isfinite(value) or value <= 0 or value >= 1
           for value in result.values()):
        raise ValueError("split ratios must be finite and between zero and one")
    if not math.isclose(sum(result.values()), 1.0, abs_tol=1e-9):
        raise ValueError("split ratios must sum to one")
    return result


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return dict(value)


def _minimum_positive_counts(
    value: int | Mapping[str, int],
) -> dict[str, int]:
    names = ("train", "validation", "test")
    if isinstance(value, Mapping):
        if set(value) != set(names):
            raise ValueError(
                "minimum_positives must contain train, validation, and test"
            )
        result = {name: int(value[name]) for name in names}
    else:
        result = {name: int(value) for name in names}
    if any(count < 1 for count in result.values()):
        raise ValueError("minimum_positives must be at least one in every split")
    return result


def _component_parameters(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    nested = value.get("parameters")
    if nested is not None:
        if not isinstance(nested, Mapping):
            raise TypeError("component parameters must be a mapping")
        return dict(nested)
    return {
        str(key): item
        for key, item in value.items()
        if key != "enabled"
    }


def _enabled(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, Mapping):
        return bool(value.get("enabled", default))
    return bool(value)


def _training_config(
    value: TrainingConfig | Mapping[str, Any] | None,
) -> TrainingConfig:
    if value is None:
        return TrainingConfig()
    if isinstance(value, TrainingConfig):
        return value
    return TrainingConfig.from_mapping(value)


def _model_layer() -> Any:
    return importlib.import_module("backend.app.ml.model")


def _is_repeatable_source(source: Any) -> bool:
    return isinstance(source, (str, Path, pd.DataFrame))


__all__ = [
    "DEFAULT_LIGHTGBM_PARAMETERS",
    "ModelBatch",
    "SplitScores",
    "TrainingConfig",
    "TrainingResult",
    "collect_feature_split",
    "collect_training_sample",
    "evaluate_stream",
    "train_from_raw",
    "validate_binary_split",
]
