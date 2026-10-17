"""Leakage-safe preprocessing for transaction feature frames.

The preprocessor is deliberately a small, auditable alternative to an opaque
``ColumnTransformer``.  ``fit`` learns every statistic and category from the
training frame only; ``transform`` is read-only and always emits the frozen
training schema.  Labels, timestamps, IDs, and case annotations are never
considered candidate model features.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import pandas as pd

from backend.app.ml.transform_features import FEATURE_COLUMNS


STATE_VERSION = 1
MISSING_CATEGORY = "__MISSING__"
UNKNOWN_CATEGORY = "__UNKNOWN__"
VALUE_PREFIX = "value:"

DEFAULT_HIGH_CARDINALITY = (
    "source_account", "destination_account", "device_info",
)
DEFAULT_LOW_CARDINALITY = ("transaction_type", "currency", "device_type")

# Explicitly documented exclusions guard against accidentally fitting on a
# newly numeric label or identifier when the input schema evolves.
NON_FEATURE_COLUMNS = frozenset({
    "event_id", "transaction_id", "transaction_fingerprint", "source_row_number",
    "dataset", "time_semantics", "event_time", "relative_time_seconds",
    "label_name", "label", "is_fraud", "is_laundering", "case_id",
    "typology", "case_detail", "pattern_detail", "case_position",
    "pattern_line_number", "pattern_id", "pattern_position", "pattern_matched",
    "laundering_case_ids", "laundering_typologies",
    "laundering_case_positions", "laundering_case_count", "analyst_outcome",
    "split",
})


@dataclass(frozen=True)
class FittedPreprocessorState:
    """JSON-safe fitted parameters and the exact output feature contract."""

    version: int
    numeric_columns: tuple[str, ...]
    numeric_medians: dict[str, float]
    high_cardinality_columns: tuple[str, ...]
    frequency_maps: dict[str, dict[str, float]]
    low_cardinality_columns: tuple[str, ...]
    one_hot_categories: dict[str, tuple[str, ...]]
    amount_center: float
    amount_scale: float
    feature_columns: tuple[str, ...]
    training_rows: int

    def to_dict(self) -> dict[str, Any]:
        """Return a canonical, strictly JSON-serializable representation."""
        return {
            "version": self.version,
            "numeric_columns": list(self.numeric_columns),
            "numeric_medians": dict(self.numeric_medians),
            "high_cardinality_columns": list(self.high_cardinality_columns),
            "frequency_maps": {
                column: dict(values)
                for column, values in self.frequency_maps.items()
            },
            "low_cardinality_columns": list(self.low_cardinality_columns),
            "one_hot_categories": {
                column: list(values)
                for column, values in self.one_hot_categories.items()
            },
            "amount_center": self.amount_center,
            "amount_scale": self.amount_scale,
            "feature_columns": list(self.feature_columns),
            "training_rows": self.training_rows,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "FittedPreprocessorState":
        """Validate and restore state produced by :meth:`to_dict`."""
        if not isinstance(raw, Mapping):
            raise TypeError("preprocessor state must be a mapping")
        if raw.get("version") != STATE_VERSION:
            raise ValueError("incompatible preprocessor state version")
        required = {
            "numeric_columns", "numeric_medians", "high_cardinality_columns",
            "frequency_maps", "low_cardinality_columns", "one_hot_categories",
            "amount_center", "amount_scale", "feature_columns", "training_rows",
        }
        missing = required - set(raw)
        if missing:
            raise ValueError(f"preprocessor state is missing: {sorted(missing)}")

        numeric = _string_tuple(raw["numeric_columns"], "numeric_columns")
        high = _string_tuple(
            raw["high_cardinality_columns"], "high_cardinality_columns"
        )
        low = _string_tuple(raw["low_cardinality_columns"],
                            "low_cardinality_columns")
        features = _string_tuple(raw["feature_columns"], "feature_columns")
        if len(features) != len(set(features)):
            raise ValueError("feature_columns contains duplicates")

        medians = _finite_float_map(raw["numeric_medians"], "numeric_medians")
        if set(medians) != set(numeric):
            raise ValueError("numeric_medians does not match numeric_columns")
        frequencies = _nested_probability_map(raw["frequency_maps"])
        if set(frequencies) != set(high):
            raise ValueError("frequency_maps does not match high-cardinality columns")
        categories = _nested_string_tuple_map(raw["one_hot_categories"])
        if set(categories) != set(low):
            raise ValueError("one_hot_categories does not match low-cardinality columns")
        for column, values in categories.items():
            if UNKNOWN_CATEGORY not in values or MISSING_CATEGORY not in values:
                raise ValueError(
                    f"{column} categories must include missing and unknown buckets"
                )
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate categories for {column}")

        expected_features = _output_columns(numeric, high, low, categories)
        if features != expected_features:
            raise ValueError(
                "feature_columns does not match the fitted preprocessing state"
            )

        center = _finite_float(raw["amount_center"], "amount_center")
        scale = _finite_float(raw["amount_scale"], "amount_scale")
        if scale <= 0:
            raise ValueError("amount_scale must be positive")
        rows = int(raw["training_rows"])
        if rows < 1:
            raise ValueError("training_rows must be positive")
        return cls(
            version=STATE_VERSION,
            numeric_columns=numeric,
            numeric_medians=medians,
            high_cardinality_columns=high,
            frequency_maps=frequencies,
            low_cardinality_columns=low,
            one_hot_categories=categories,
            amount_center=center,
            amount_scale=scale,
            feature_columns=features,
            training_rows=rows,
        )


class FeaturePreprocessor:
    """Fit-on-train numeric, frequency, and one-hot transformations.

    Numeric model inputs are restricted to the causal ``FEATURE_COLUMNS`` by
    default.  Every numeric column receives median imputation and an explicit
    missing indicator.  The imputed ``amount`` is retained and an additional
    z-scaled copy, ``amount_scaled``, is emitted.  Account identifiers use
    frequency encoding; bounded transaction attributes use one-hot encoding.
    """

    def __init__(
        self,
        *,
        numeric_columns: Sequence[str] | None = None,
        high_cardinality_columns: Sequence[str] = DEFAULT_HIGH_CARDINALITY,
        low_cardinality_columns: Sequence[str] = DEFAULT_LOW_CARDINALITY,
        max_one_hot_categories: int = 64,
    ) -> None:
        self.requested_numeric_columns = (
            None if numeric_columns is None
            else _unique_strings(numeric_columns, "numeric_columns")
        )
        self.requested_high_cardinality_columns = _unique_strings(
            high_cardinality_columns, "high_cardinality_columns"
        )
        self.requested_low_cardinality_columns = _unique_strings(
            low_cardinality_columns, "low_cardinality_columns"
        )
        overlap = (set(self.requested_high_cardinality_columns)
                   & set(self.requested_low_cardinality_columns))
        if overlap:
            raise ValueError(
                f"categorical columns cannot be both high and low cardinality: "
                f"{sorted(overlap)}"
            )
        self.max_one_hot_categories = int(max_one_hot_categories)
        if self.max_one_hot_categories < 1:
            raise ValueError("max_one_hot_categories must be at least one")
        self.state: FittedPreprocessorState | None = None

    @property
    def is_fitted(self) -> bool:
        return self.state is not None

    @property
    def feature_columns(self) -> tuple[str, ...]:
        if self.state is None:
            raise RuntimeError("preprocessor has not been fitted")
        return self.state.feature_columns

    def fit(self, train: pd.DataFrame) -> "FeaturePreprocessor":
        """Learn all preprocessing state from ``train`` and no other split."""
        _validate_frame(train, require_rows=True)
        numeric = self._numeric_columns(train)
        high = tuple(column for column in self.requested_high_cardinality_columns
                     if column in train.columns)
        low = tuple(column for column in self.requested_low_cardinality_columns
                    if column in train.columns)
        if not numeric and not high and not low:
            raise ValueError("training frame has no configured model inputs")

        medians: dict[str, float] = {}
        numeric_training: dict[str, pd.Series] = {}
        for column in numeric:
            values = _numeric_series(train[column])
            finite = values.dropna()
            median = float(finite.median()) if not finite.empty else 0.0
            medians[column] = median
            numeric_training[column] = values.fillna(median)

        frequencies: dict[str, dict[str, float]] = {}
        for column in high:
            values = [_category(value) for value in train[column].tolist()]
            counts = Counter(values)
            frequencies[column] = {
                key: count / len(train)
                for key, count in sorted(counts.items())
            }

        categories: dict[str, tuple[str, ...]] = {}
        for column in low:
            values = [_category(value) for value in train[column].tolist()]
            counts = Counter(value for value in values
                             if value != MISSING_CATEGORY)
            # Category selection is bounded and based solely on training data.
            most_common = sorted(
                counts,
                key=lambda value: (-counts[value], value),
            )[:self.max_one_hot_categories]
            categories[column] = tuple(
                [*sorted(most_common), MISSING_CATEGORY, UNKNOWN_CATEGORY]
            )

        if "amount" in numeric_training:
            amount_values = numeric_training["amount"].to_numpy(dtype=float)
            center = float(amount_values.mean())
            scale = float(amount_values.std(ddof=0))
            if not math.isfinite(scale) or scale <= 0:
                scale = 1.0
        else:
            center, scale = 0.0, 1.0

        output_columns = _output_columns(numeric, high, low, categories)
        self.state = FittedPreprocessorState(
            version=STATE_VERSION,
            numeric_columns=numeric,
            numeric_medians=medians,
            high_cardinality_columns=high,
            frequency_maps=frequencies,
            low_cardinality_columns=low,
            one_hot_categories=categories,
            amount_center=center,
            amount_scale=scale,
            feature_columns=output_columns,
            training_rows=len(train),
        )
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply frozen training state without mutating or extending it."""
        _validate_frame(frame, require_rows=False)
        state = self._require_state()
        result: dict[str, pd.Series | np.ndarray] = {}

        for column in state.numeric_columns:
            raw = (_numeric_series(frame[column]) if column in frame.columns
                   else pd.Series(np.nan, index=frame.index, dtype=float))
            missing = raw.isna()
            imputed = raw.fillna(state.numeric_medians[column]).astype(float)
            result[column] = imputed
            result[f"{column}__missing"] = missing.astype("int8")
            if column == "amount":
                result["amount_scaled"] = (
                    (imputed - state.amount_center) / state.amount_scale
                ).astype(float)

        for column in state.high_cardinality_columns:
            raw = (frame[column] if column in frame.columns
                   else pd.Series(pd.NA, index=frame.index, dtype="object"))
            mapping = state.frequency_maps[column]
            result[f"{column}__frequency"] = pd.Series(
                (_category(value) for value in raw.tolist()), index=frame.index
            ).map(mapping).fillna(0.0).astype(float)

        for column in state.low_cardinality_columns:
            raw = (frame[column] if column in frame.columns
                   else pd.Series(pd.NA, index=frame.index, dtype="object"))
            allowed = state.one_hot_categories[column]
            known = set(allowed) - {UNKNOWN_CATEGORY}
            normalized = pd.Series(
                (_category(value) for value in raw.tolist()), index=frame.index
            )
            normalized = normalized.where(normalized.isin(known), UNKNOWN_CATEGORY)
            for category in allowed:
                result[_one_hot_name(column, category)] = (
                    normalized == category
                ).astype("int8")

        transformed = pd.DataFrame(result, index=frame.index)
        transformed = transformed.loc[:, list(state.feature_columns)]
        transformed.attrs.update({
            "feature_columns": list(state.feature_columns),
            "preprocessor_state_version": state.version,
            "fit_on_training_rows": state.training_rows,
        })
        return transformed

    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train).transform(train)

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration and fitted state with no Python objects."""
        state = self._require_state()
        return {
            "kind": "capacity-aware-aml-feature-preprocessor",
            "state": state.to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "FeaturePreprocessor":
        if not isinstance(raw, Mapping) or raw.get("kind") != (
            "capacity-aware-aml-feature-preprocessor"
        ):
            raise ValueError("not a supported feature preprocessor document")
        state = FittedPreprocessorState.from_dict(raw.get("state", {}))
        instance = cls(
            numeric_columns=state.numeric_columns,
            high_cardinality_columns=state.high_cardinality_columns,
            low_cardinality_columns=state.low_cardinality_columns,
            max_one_hot_categories=max(
                1,
                *(len(values) - 2 for values in state.one_hot_categories.values()),
            ),
        )
        instance.state = state
        return instance

    def save(self, path: str | Path) -> Path:
        """Write fitted state as deterministic strict JSON."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                self.to_dict(), indent=2, sort_keys=True, allow_nan=False
            ) + "\n",
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> "FeaturePreprocessor":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def _numeric_columns(self, train: pd.DataFrame) -> tuple[str, ...]:
        requested = (tuple(FEATURE_COLUMNS) if self.requested_numeric_columns is None
                     else self.requested_numeric_columns)
        forbidden = set(requested) & NON_FEATURE_COLUMNS
        if forbidden:
            raise ValueError(
                f"metadata/label columns cannot be model features: {sorted(forbidden)}"
            )
        return tuple(column for column in requested if column in train.columns)

    def _require_state(self) -> FittedPreprocessorState:
        if self.state is None:
            raise RuntimeError("preprocessor has not been fitted")
        return self.state


def _output_columns(
    numeric: Sequence[str],
    high: Sequence[str],
    low: Sequence[str],
    categories: Mapping[str, Sequence[str]],
) -> tuple[str, ...]:
    answer: list[str] = []
    for column in numeric:
        answer.extend((column, f"{column}__missing"))
        if column == "amount":
            answer.append("amount_scaled")
    answer.extend(f"{column}__frequency" for column in high)
    for column in low:
        answer.extend(_one_hot_name(column, category)
                      for category in categories[column])
    if len(answer) != len(set(answer)):
        raise ValueError("preprocessed feature names are not unique")
    return tuple(answer)


def _category(value: Any) -> str:
    if value is None:
        return MISSING_CATEGORY
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return MISSING_CATEGORY
    return VALUE_PREFIX + str(value)


def _one_hot_name(column: str, category: str) -> str:
    return f"{column}__one_hot={quote(category, safe='')}"


def _numeric_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    return numeric.where(np.isfinite(numeric), np.nan)


def _validate_frame(frame: pd.DataFrame, *, require_rows: bool) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if frame.columns.has_duplicates:
        raise ValueError("frame has duplicate columns")
    if require_rows and frame.empty:
        raise ValueError("training frame must contain at least one row")


def _unique_strings(values: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of column names")
    result = tuple(str(value) for value in values)
    if any(not value for value in result):
        raise ValueError(f"{name} cannot contain empty names")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} cannot contain duplicates")
    return result


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a JSON string list")
    return tuple(value)


def _finite_float(value: Any, name: str) -> float:
    try:
        answer = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(answer):
        raise ValueError(f"{name} must be finite")
    return answer


def _finite_float_map(value: Any, name: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return {str(key): _finite_float(item, f"{name}.{key}")
            for key, item in value.items()}


def _nested_probability_map(value: Any) -> dict[str, dict[str, float]]:
    if not isinstance(value, Mapping):
        raise ValueError("frequency_maps must be a mapping")
    answer: dict[str, dict[str, float]] = {}
    for column, raw_mapping in value.items():
        mapping = _finite_float_map(raw_mapping, f"frequency_maps.{column}")
        if any(probability < 0 or probability > 1
               for probability in mapping.values()):
            raise ValueError("frequencies must be between zero and one")
        answer[str(column)] = mapping
    return answer


def _nested_string_tuple_map(value: Any) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, Mapping):
        raise ValueError("one_hot_categories must be a mapping")
    return {str(column): _string_tuple(items, f"categories.{column}")
            for column, items in value.items()}


__all__ = [
    "DEFAULT_HIGH_CARDINALITY", "DEFAULT_LOW_CARDINALITY",
    "FeaturePreprocessor", "FittedPreprocessorState", "MISSING_CATEGORY",
    "NON_FEATURE_COLUMNS", "STATE_VERSION", "UNKNOWN_CATEGORY",
]
