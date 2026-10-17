"""Bounded-memory features and chronological splits for raw AMLworld data.

The full Sprint 1 temporal feature transformer intentionally retains account
history.  That is the right semantic contract for those features, but the
history can be too large for a small development machine when all 5 million
AMLworld rows are used.  This module provides a compact, stateless fallback
for model training directly from the governed transaction CSV.

Two passes are used.  The first pass stores only one aggregate per timestamp
and chooses 70/15/15 boundaries without ever dividing an equal-time group.
The second pass constructs numeric features a chunk at a time.  The physical
CSV does not need to be ordered: split membership depends only on the frozen
timestamp cutoffs.  No label is used to construct a predictor.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import math
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import pandas as pd


SPLIT_NAMES = ("train", "validation", "test")
RAW_FEATURE_SET_VERSION = "raw-baseline-v1"
DEFAULT_SPLIT_RATIOS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}

# All model inputs are numeric and stateless.  Stable hashes deliberately
# avoid a fitted vocabulary, which keeps validation/test transformation
# bounded and makes the same contract usable by an online scorer.
RAW_FEATURE_COLUMNS = (
    "amount",
    "amount_received",
    "signed_log_amount",
    "signed_log_amount_received",
    "amount_absolute_difference",
    "amount_paid_to_received_ratio",
    "amount_missing",
    "amount_received_missing",
    "same_bank",
    "same_account",
    "currency_mismatch",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "is_weekend",
    "source_bank_hash",
    "destination_bank_hash",
    "source_account_hash",
    "destination_account_hash",
    "receiving_currency_hash",
    "payment_currency_hash",
    "payment_format_hash",
)

RAW_METADATA_COLUMNS = (
    "event_id",
    "source_row_number",
    "event_time",
    "transaction_fingerprint",
    "transaction_amount",
    "label",
    "split",
)

HASH_BUCKETS = 4096
_FINGERPRINT_SEPARATOR = "\x1f"

_ALIASES: dict[str, tuple[str, ...]] = {
    "timestamp": ("Timestamp", "event_time", "timestamp", "Date Time"),
    "label": ("Is Laundering", "label", "is_laundering", "laundering"),
    "source_bank": ("From Bank", "source_bank", "Sender Bank"),
    "source_account": ("Account", "source_account", "From Account"),
    "destination_bank": ("To Bank", "destination_bank", "Receiver Bank"),
    "destination_account": (
        "Account.1", "destination_account", "To Account", "Receiver Account",
    ),
    "amount_received": (
        "Amount Received", "amount_received", "Receiving Amount",
    ),
    "amount_paid": ("Amount Paid", "amount", "amount_paid"),
    "receiving_currency": (
        "Receiving Currency", "receiving_currency",
    ),
    "payment_currency": ("Payment Currency", "currency", "payment_currency"),
    "payment_format": (
        "Payment Format", "transaction_type", "payment_format", "type",
    ),
    "event_id": ("event_id", "transaction_id"),
    "source_row_number": ("source_row_number",),
}

RawSource: TypeAlias = str | Path | pd.DataFrame | Iterable[pd.DataFrame]


@dataclass(frozen=True)
class SplitBoundaries:
    """Frozen timestamp-group boundaries and audited class counts."""

    validation_start: pd.Timestamp
    test_start: pd.Timestamp
    total_rows: int
    timestamp_groups: int
    target_ratios: tuple[float, float, float]
    rows: tuple[int, int, int]
    positives: tuple[int, int, int]
    minimum_event_time: pd.Timestamp
    maximum_event_time: pd.Timestamp

    @property
    def negatives(self) -> tuple[int, int, int]:
        return tuple(
            int(rows - positives)
            for rows, positives in zip(self.rows, self.positives)
        )

    def count(self, split: str) -> int:
        """Return the full, unsampled row count for ``split``."""
        return self.rows[_split_index(split)]

    def positive_count(self, split: str) -> int:
        return self.positives[_split_index(split)]

    def negative_count(self, split: str) -> int:
        return self.negatives[_split_index(split)]

    def assign(self, timestamps: Sequence[Any] | pd.Series) -> np.ndarray:
        """Assign parsed timestamps using the frozen half-open periods."""
        parsed = _parse_timestamps(pd.Series(timestamps, copy=False))
        result = np.full(len(parsed), "test", dtype=object)
        result[(parsed < self.validation_start).to_numpy()] = "train"
        validation = (
            (parsed >= self.validation_start) & (parsed < self.test_start)
        ).to_numpy()
        result[validation] = "validation"
        return result

    def to_dict(self) -> dict[str, Any]:
        """Return strict-JSON-friendly split provenance."""
        return {
            "strategy": "chronological_timestamp_groups",
            "validation_starts_at": self.validation_start.isoformat(),
            "test_starts_at": self.test_start.isoformat(),
            "minimum_event_time": self.minimum_event_time.isoformat(),
            "maximum_event_time": self.maximum_event_time.isoformat(),
            "total_rows": self.total_rows,
            "timestamp_groups": self.timestamp_groups,
            "target_ratios": dict(zip(SPLIT_NAMES, self.target_ratios)),
            "splits": {
                name: {
                    "rows": self.rows[index],
                    "positives": self.positives[index],
                    "negatives": self.negatives[index],
                    "actual_ratio": self.rows[index] / self.total_rows,
                }
                for index, name in enumerate(SPLIT_NAMES)
            },
        }


@dataclass(frozen=True)
class RawFeatureChunk:
    """A model matrix chunk plus aligned metadata and labels."""

    features: pd.DataFrame
    labels: np.ndarray
    metadata: pd.DataFrame
    split: str

    def __post_init__(self) -> None:
        rows = len(self.features)
        if len(self.labels) != rows or len(self.metadata) != rows:
            raise ValueError("raw feature chunk components are not row-aligned")


def scan_timestamp_groups(
    source: RawSource,
    *,
    chunk_size: int = 100_000,
) -> pd.DataFrame:
    """Stream ``source`` and aggregate row/positive counts by event time.

    The returned frame is small (one row per distinct timestamp), sorted, and
    has the columns ``event_time``, ``rows``, and ``positives``.  Labels are
    validated as binary here so a malformed source fails before fitting.
    """
    _positive_int(chunk_size, "chunk_size")
    pieces: list[pd.DataFrame] = []
    observed_rows = 0
    for raw in _iter_source_chunks(source, chunk_size=chunk_size):
        timestamp_column = _required_column(raw, "timestamp")
        label_column = _required_column(raw, "label")
        timestamps = _parse_timestamps(raw[timestamp_column])
        labels = _binary_labels(raw[label_column], name="raw source")
        grouped = pd.DataFrame({"event_time": timestamps, "label": labels}).groupby(
            "event_time", sort=False, observed=True
        )["label"].agg(rows="size", positives="sum").reset_index()
        pieces.append(grouped)
        observed_rows += len(raw)

    if observed_rows == 0:
        raise ValueError("raw source contains no data rows")
    groups = pd.concat(pieces, ignore_index=True).groupby(
        "event_time", sort=True, observed=True, as_index=False
    ).agg(rows=("rows", "sum"), positives=("positives", "sum"))
    groups["rows"] = groups["rows"].astype("int64")
    groups["positives"] = groups["positives"].astype("int64")
    if int(groups["rows"].sum()) != observed_rows:
        raise RuntimeError("timestamp aggregation did not preserve every source row")
    return groups.sort_values("event_time", kind="mergesort").reset_index(drop=True)


def timestamp_group_split(
    timestamp_groups: pd.DataFrame,
    *,
    ratios: Mapping[str, float] | None = None,
    minimum_positives: int | Mapping[str, int] = 1,
    require_both_classes: bool = True,
) -> SplitBoundaries:
    """Choose deterministic 70/15/15 row targets between timestamp groups.

    The nominal boundaries use the same semantics as selecting the row at the
    cumulative target and placing its entire timestamp group in the later
    split.  Unique timestamps therefore give exact 70/15/15 row targets;
    repeated timestamps may move a boundary earlier but can never leak across
    adjacent periods.
    """
    if not isinstance(timestamp_groups, pd.DataFrame):
        raise TypeError("timestamp_groups must be a pandas DataFrame")
    required = {"event_time", "rows", "positives"}
    missing = required - set(timestamp_groups.columns)
    if missing:
        raise ValueError(f"timestamp groups are missing columns: {sorted(missing)}")
    proportions = _validated_ratios(ratios or DEFAULT_SPLIT_RATIOS)
    minimum_by_split = _minimum_positive_counts(minimum_positives)

    groups = timestamp_groups.loc[:, ["event_time", "rows", "positives"]].copy()
    groups["event_time"] = _parse_timestamps(groups["event_time"])
    groups = groups.sort_values("event_time", kind="mergesort").reset_index(drop=True)
    if groups["event_time"].duplicated().any():
        groups = groups.groupby("event_time", sort=True, as_index=False).agg(
            rows=("rows", "sum"), positives=("positives", "sum")
        )
    rows = _nonnegative_integer_array(groups["rows"], "rows")
    positives = _nonnegative_integer_array(groups["positives"], "positives")
    if np.any(rows == 0):
        raise ValueError("timestamp group row counts must be positive")
    if np.any(positives > rows):
        raise ValueError("timestamp group positives cannot exceed rows")
    total = int(rows.sum())
    if total < 3 or len(groups) < 3:
        raise ValueError("at least three rows and three timestamps are required")

    train_target = min(max(1, int(total * proportions["train"])), total - 2)
    validation_end_target = max(
        train_target + 1,
        int(total * (proportions["train"] + proportions["validation"])),
    )
    validation_end_target = min(validation_end_target, total - 1)
    cumulative = np.cumsum(rows, dtype=np.int64)
    train_group = int(np.searchsorted(cumulative, train_target, side="right"))
    test_group = int(
        np.searchsorted(cumulative, validation_end_target, side="right")
    )
    if train_group < 1 or test_group <= train_group or test_group >= len(groups):
        raise ValueError(
            "timestamp groups are too coarse for three non-empty chronological splits"
        )

    split_rows = (
        int(rows[:train_group].sum()),
        int(rows[train_group:test_group].sum()),
        int(rows[test_group:].sum()),
    )
    split_positives = (
        int(positives[:train_group].sum()),
        int(positives[train_group:test_group].sum()),
        int(positives[test_group:].sum()),
    )
    if sum(split_rows) != total or any(value < 1 for value in split_rows):
        raise RuntimeError("timestamp boundaries did not preserve non-empty splits")
    if require_both_classes:
        for name, count, positive in zip(SPLIT_NAMES, split_rows, split_positives):
            _validate_counts(
                count,
                positive,
                name,
                minimum_positives=minimum_by_split[name],
                minimum_negatives=1,
            )

    event_times = groups["event_time"]
    return SplitBoundaries(
        validation_start=pd.Timestamp(event_times.iloc[train_group]),
        test_start=pd.Timestamp(event_times.iloc[test_group]),
        total_rows=total,
        timestamp_groups=len(groups),
        target_ratios=tuple(proportions[name] for name in SPLIT_NAMES),
        rows=split_rows,
        positives=split_positives,
        minimum_event_time=pd.Timestamp(event_times.iloc[0]),
        maximum_event_time=pd.Timestamp(event_times.iloc[-1]),
    )


def plan_chronological_splits(
    source: RawSource,
    *,
    chunk_size: int = 100_000,
    ratios: Mapping[str, float] | None = None,
    minimum_positives: int | Mapping[str, int] = 1,
    require_both_classes: bool = True,
) -> SplitBoundaries:
    """Scan a repeatable source and freeze its timestamp-group split plan."""
    return timestamp_group_split(
        scan_timestamp_groups(source, chunk_size=chunk_size),
        ratios=ratios,
        minimum_positives=minimum_positives,
        require_both_classes=require_both_classes,
    )


def transform_raw_chunk(
    raw: pd.DataFrame,
    *,
    split: str | Sequence[str] = "train",
    feature_columns: Sequence[str] | None = None,
) -> RawFeatureChunk:
    """Construct compact label-blind numeric features from one raw chunk."""
    if not isinstance(raw, pd.DataFrame):
        raise TypeError("raw must be a pandas DataFrame")
    if raw.empty:
        columns = _selected_features(feature_columns)
        empty_metadata = pd.DataFrame(columns=RAW_METADATA_COLUMNS)
        return RawFeatureChunk(
            pd.DataFrame(columns=columns),
            np.empty(0, dtype=np.uint8),
            empty_metadata,
            str(split) if isinstance(split, str) else "mixed",
        )
    timestamp_name = _required_column(raw, "timestamp")
    label_name = _required_column(raw, "label")
    event_time = _parse_timestamps(raw[timestamp_name]).reset_index(drop=True)
    labels = _binary_labels(raw[label_name], name="raw feature chunk")

    source_rows = _source_row_numbers(raw)
    split_values = _split_values(split, len(raw))
    event_ids = _optional_series(raw, "event_id", default=None)
    fingerprints = _positive_transaction_fingerprints(raw, labels)
    if event_ids is None:
        event_ids = pd.Series(
            [f"amlworld-row-{int(row):09d}" for row in source_rows],
            dtype="string",
        )
    else:
        event_ids = event_ids.reset_index(drop=True).astype("string")

    source_bank = _clean_text(_optional_series(raw, "source_bank"))
    destination_bank = _clean_text(_optional_series(raw, "destination_bank"))
    source_account_raw = _clean_text(_optional_series(raw, "source_account"))
    destination_account_raw = _clean_text(
        _optional_series(raw, "destination_account")
    )
    source_account = source_bank + "\x1f" + source_account_raw
    destination_account = destination_bank + "\x1f" + destination_account_raw
    source_account = source_account.where(
        source_bank.ne("") | source_account_raw.ne(""), ""
    )
    destination_account = destination_account.where(
        destination_bank.ne("") | destination_account_raw.ne(""), ""
    )
    receiving_currency = _clean_text(
        _optional_series(raw, "receiving_currency")
    )
    payment_currency = _clean_text(_optional_series(raw, "payment_currency"))
    payment_format = _clean_text(_optional_series(raw, "payment_format"))

    amount_paid = _numeric(_optional_series(raw, "amount_paid"))
    amount_received = _numeric(_optional_series(raw, "amount_received"))
    paid_missing = amount_paid.isna()
    received_missing = amount_received.isna()
    paid = amount_paid.fillna(0.0).to_numpy(dtype=np.float64)
    received = amount_received.fillna(0.0).to_numpy(dtype=np.float64)
    difference = np.abs(paid - received)
    denominator = np.maximum(np.abs(received), 1e-6)
    ratio = np.clip(paid / denominator, -1_000_000.0, 1_000_000.0)

    hour = (event_time.dt.hour + event_time.dt.minute / 60.0).to_numpy()
    weekday = event_time.dt.dayofweek.to_numpy()
    features = pd.DataFrame({
        "amount": paid,
        "amount_received": received,
        "signed_log_amount": np.sign(paid) * np.log1p(np.abs(paid)),
        "signed_log_amount_received": (
            np.sign(received) * np.log1p(np.abs(received))
        ),
        "amount_absolute_difference": difference,
        "amount_paid_to_received_ratio": ratio,
        "amount_missing": paid_missing.to_numpy(dtype=np.int8),
        "amount_received_missing": received_missing.to_numpy(dtype=np.int8),
        "same_bank": (
            source_bank.ne("") & source_bank.eq(destination_bank)
        ).to_numpy(dtype=np.int8),
        "same_account": (
            source_account_raw.ne("")
            & destination_account_raw.ne("")
            & source_account.eq(destination_account)
        ).to_numpy(dtype=np.int8),
        "currency_mismatch": (
            receiving_currency.ne("")
            & payment_currency.ne("")
            & receiving_currency.ne(payment_currency)
        ).to_numpy(dtype=np.int8),
        "hour_sin": np.sin(2.0 * math.pi * hour / 24.0),
        "hour_cos": np.cos(2.0 * math.pi * hour / 24.0),
        "weekday_sin": np.sin(2.0 * math.pi * weekday / 7.0),
        "weekday_cos": np.cos(2.0 * math.pi * weekday / 7.0),
        "is_weekend": (weekday >= 5).astype(np.int8),
        "source_bank_hash": _stable_category_hash(source_bank),
        "destination_bank_hash": _stable_category_hash(destination_bank),
        "source_account_hash": _stable_category_hash(source_account),
        "destination_account_hash": _stable_category_hash(destination_account),
        "receiving_currency_hash": _stable_category_hash(receiving_currency),
        "payment_currency_hash": _stable_category_hash(payment_currency),
        "payment_format_hash": _stable_category_hash(payment_format),
    })
    features = features.astype("float32")
    selected = _selected_features(feature_columns)
    features = features.loc[:, list(selected)]
    features.attrs["feature_set"] = RAW_FEATURE_SET_VERSION
    features.attrs["feature_columns"] = list(selected)
    metadata = pd.DataFrame({
        "event_id": event_ids.to_numpy(),
        "source_row_number": source_rows,
        "event_time": event_time.to_numpy(),
        "transaction_fingerprint": fingerprints,
        "transaction_amount": paid,
        "label": labels,
        "split": split_values,
    })
    chunk_name = str(split_values[0]) if np.all(split_values == split_values[0]) else "mixed"
    return RawFeatureChunk(features, labels, metadata, chunk_name)


def iter_raw_feature_chunks(
    source: RawSource,
    boundaries: SplitBoundaries,
    *,
    split: str | None = None,
    chunk_size: int = 100_000,
    feature_columns: Sequence[str] | None = None,
) -> Iterator[RawFeatureChunk]:
    """Yield bounded numeric chunks for one full, unsampled split.

    If ``split`` is ``None``, each yielded chunk may contain mixed split names.
    Supplying a split is recommended for training and evaluation and guarantees
    that every emitted chunk has that single split.
    """
    if not isinstance(boundaries, SplitBoundaries):
        raise TypeError("boundaries must be SplitBoundaries")
    _positive_int(chunk_size, "chunk_size")
    if split is not None:
        _split_index(split)
    for raw in _iter_source_chunks(source, chunk_size=chunk_size):
        timestamp_name = _required_column(raw, "timestamp")
        assignments = boundaries.assign(raw[timestamp_name])
        if split is None:
            selected = raw
            selected_assignments: str | Sequence[str] = assignments
        else:
            mask = assignments == split
            if not bool(mask.any()):
                continue
            selected = raw.loc[mask].reset_index(drop=True)
            selected_assignments = split
        yield transform_raw_chunk(
            selected,
            split=selected_assignments,
            feature_columns=feature_columns,
        )


def deterministic_sample_keys(
    row_ids: Sequence[Any] | pd.Series | np.ndarray,
    *,
    seed: int = 20261007,
) -> np.ndarray:
    """Return stable unsigned pseudo-random keys independent of chunking."""
    values = pd.Series(row_ids, copy=False)
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all() and np.all(np.equal(numeric, np.floor(numeric))):
        base = numeric.to_numpy(dtype=np.uint64)
    else:
        base = pd.util.hash_pandas_object(
            values.astype("string"), index=False, categorize=True
        ).to_numpy(dtype=np.uint64)
    mask = (1 << 64) - 1
    seed_word = np.uint64(int(seed) & mask)
    with np.errstate(over="ignore"):
        mixed = base + seed_word + np.uint64(0x9E3779B97F4A7C15)
        mixed = (mixed ^ (mixed >> np.uint64(30))) * np.uint64(
            0xBF58476D1CE4E5B9
        )
        mixed = (mixed ^ (mixed >> np.uint64(27))) * np.uint64(
            0x94D049BB133111EB
        )
        mixed = mixed ^ (mixed >> np.uint64(31))
    return mixed


def deterministic_negative_mask(
    labels: Sequence[Any] | pd.Series | np.ndarray,
    row_ids: Sequence[Any] | pd.Series | np.ndarray,
    *,
    negative_keep_probability: float,
    seed: int = 20261007,
) -> np.ndarray:
    """Keep all positives and a stable Bernoulli-like subset of negatives."""
    binary = _binary_labels(pd.Series(labels, copy=False), name="sample labels")
    if len(binary) != len(row_ids):
        raise ValueError("labels and row_ids must have equal length")
    probability = float(negative_keep_probability)
    if not math.isfinite(probability) or probability < 0.0 or probability > 1.0:
        raise ValueError("negative_keep_probability must be between zero and one")
    threshold = int(probability * ((1 << 64) - 1))
    keys = deterministic_sample_keys(row_ids, seed=seed)
    return (binary == 1) | (keys <= np.uint64(threshold))


def deterministic_undersample(
    frame: pd.DataFrame,
    *,
    label_column: str = "label",
    row_id_column: str = "source_row_number",
    negative_to_positive_ratio: float = 20.0,
    seed: int = 20261007,
) -> pd.DataFrame:
    """Keep every positive and exactly the lowest-hash training negatives."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if label_column not in frame:
        raise ValueError(f"missing label column: {label_column}")
    if row_id_column not in frame:
        raise ValueError(f"missing row ID column: {row_id_column}")
    ratio = float(negative_to_positive_ratio)
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("negative_to_positive_ratio must be positive and finite")
    labels = _binary_labels(frame[label_column], name="training sample")
    positive_mask = labels == 1
    positive_count = int(positive_mask.sum())
    negative_count = len(frame) - positive_count
    _validate_counts(
        len(frame), positive_count, "training sample",
        minimum_positives=1, minimum_negatives=1,
    )
    requested = min(negative_count, int(math.ceil(positive_count * ratio)))
    negatives = frame.loc[~positive_mask].copy()
    negatives["__sample_key"] = deterministic_sample_keys(
        negatives[row_id_column], seed=seed
    )
    negatives = negatives.nsmallest(
        requested, ["__sample_key", row_id_column], keep="first"
    ).drop(columns="__sample_key")
    sampled = pd.concat([frame.loc[positive_mask], negatives], ignore_index=True)
    keys = deterministic_sample_keys(sampled[row_id_column], seed=seed)
    return sampled.assign(__order=keys).sort_values(
        ["__order", row_id_column], kind="mergesort"
    ).drop(columns="__order").reset_index(drop=True)


def _iter_source_chunks(source: RawSource, *, chunk_size: int) -> Iterator[pd.DataFrame]:
    next_source_row = 2
    if isinstance(source, pd.DataFrame):
        iterator: Iterable[pd.DataFrame] = (
            source.iloc[start : start + chunk_size].copy()
            for start in range(0, len(source), chunk_size)
        )
    elif isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(path)
        iterator = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
            chunksize=chunk_size,
            encoding="utf-8-sig",
        )
    elif isinstance(source, Iterable):
        iterator = source
    else:
        raise TypeError("source must be a CSV path, DataFrame, or frame iterable")

    for value in iterator:
        if not isinstance(value, pd.DataFrame):
            raise TypeError("source iterable must yield pandas DataFrames")
        raw = value.copy()
        if raw.empty:
            continue
        source_row_name = _find_column(raw, "source_row_number")
        if source_row_name is None:
            raw["source_row_number"] = np.arange(
                next_source_row, next_source_row + len(raw), dtype=np.int64
            )
        else:
            converted = pd.to_numeric(raw[source_row_name], errors="coerce")
            if converted.isna().any() or not np.all(converted == np.floor(converted)):
                raise ValueError("source_row_number must contain integers")
            raw["source_row_number"] = converted.astype("int64")
        next_source_row += len(raw)
        yield raw


def _source_row_numbers(raw: pd.DataFrame) -> np.ndarray:
    name = _find_column(raw, "source_row_number")
    if name is None:
        return np.arange(2, len(raw) + 2, dtype=np.int64)
    values = pd.to_numeric(raw[name], errors="coerce")
    if values.isna().any() or not np.all(values == np.floor(values)):
        raise ValueError("source_row_number must contain integers")
    return values.to_numpy(dtype=np.int64)


def _parse_timestamps(values: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(values.dtype):
        parsed = pd.to_datetime(values, errors="coerce", utc=True)
    else:
        parsed = pd.to_datetime(values, errors="coerce", utc=True, format="mixed")
    if parsed.isna().any():
        bad = values.loc[parsed.isna()].iloc[0]
        raise ValueError(f"invalid event timestamp: {bad!r}")
    return parsed.astype("datetime64[ns, UTC]")


def _binary_labels(values: pd.Series, *, name: str) -> np.ndarray:
    normalized = values.astype("string").str.strip().str.casefold().replace({
        "true": "1", "false": "0", "yes": "1", "no": "0",
    })
    numeric = pd.to_numeric(normalized, errors="coerce")
    if numeric.isna().any() or not numeric.isin([0, 1]).all():
        invalid = values.loc[numeric.isna() | ~numeric.isin([0, 1])].iloc[0]
        raise ValueError(f"{name} contains a non-binary label: {invalid!r}")
    return numeric.to_numpy(dtype=np.uint8)


def _numeric(values: pd.Series | None) -> pd.Series:
    if values is None:
        return pd.Series(dtype=float)
    converted = pd.to_numeric(values, errors="coerce").astype(float)
    return converted.where(np.isfinite(converted), np.nan).reset_index(drop=True)


def _clean_text(values: pd.Series | None) -> pd.Series:
    if values is None:
        raise RuntimeError("internal optional series requires a row count")
    return values.astype("string").fillna("").str.strip().reset_index(drop=True)


def _stable_category_hash(values: pd.Series) -> np.ndarray:
    missing = values.eq("").to_numpy()
    hashed = pd.util.hash_pandas_object(
        values, index=False, categorize=True
    ).to_numpy(dtype=np.uint64)
    # Stable integer buckets let LightGBM treat these columns as categorical.
    # Zero is reserved for missing; non-empty values use 1..HASH_BUCKETS.
    result = (hashed % np.uint64(HASH_BUCKETS)).astype(np.float64) + 1.0
    result[missing] = 0.0
    return result.astype(np.float32)


def _positive_transaction_fingerprints(
    raw: pd.DataFrame,
    labels: np.ndarray,
) -> np.ndarray:
    """Return AMLworld fingerprints for positives only.

    Fingerprints are evaluation-only ground-truth join keys.  Computing them
    only for positives keeps the full holdout metadata compact, and the values
    never enter :data:`RAW_FEATURE_COLUMNS`.
    """
    result = np.full(len(raw), None, dtype=object)
    positive_positions = np.flatnonzero(labels == 1)
    if not len(positive_positions):
        return result
    keys = (
        "timestamp",
        "source_bank",
        "source_account",
        "destination_bank",
        "destination_account",
        "amount_received",
        "receiving_currency",
        "amount_paid",
        "payment_currency",
        "payment_format",
        "label",
    )
    names = [_find_column(raw, key) for key in keys]
    if any(name is None for name in names):
        return result
    columns = [str(name) for name in names]
    selected = raw.iloc[positive_positions].loc[:, columns]
    for position, values in zip(
        positive_positions,
        selected.itertuples(index=False, name=None),
    ):
        canonical = _FINGERPRINT_SEPARATOR.join(
            "" if value is None else str(value).strip() for value in values
        )
        result[position] = (
            "amlworld-tx-" + sha256(canonical.encode("utf-8")).hexdigest()
        )
    return result


def _optional_series(
    frame: pd.DataFrame,
    key: str,
    *,
    default: Any = "",
) -> pd.Series | None:
    name = _find_column(frame, key)
    if name is not None:
        return frame[name]
    if default is None:
        return None
    return pd.Series([default] * len(frame), index=frame.index, dtype="object")


def _required_column(frame: pd.DataFrame, key: str) -> str:
    name = _find_column(frame, key)
    if name is None:
        raise ValueError(
            f"raw AMLworld data is missing {key}; expected one of {_ALIASES[key]}"
        )
    return name


def _find_column(frame: pd.DataFrame, key: str) -> str | None:
    exact = {str(column): str(column) for column in frame.columns}
    for alias in _ALIASES[key]:
        if alias in exact:
            return exact[alias]
    normalized = {_normalize(column): str(column) for column in frame.columns}
    for alias in _ALIASES[key]:
        match = normalized.get(_normalize(alias))
        if match is not None:
            return match
    return None


def _normalize(value: Any) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())


def _selected_features(feature_columns: Sequence[str] | None) -> tuple[str, ...]:
    if feature_columns is None:
        return RAW_FEATURE_COLUMNS
    if isinstance(feature_columns, (str, bytes)) or not isinstance(
        feature_columns, Sequence
    ):
        raise TypeError("feature_columns must be a sequence")
    selected = tuple(map(str, feature_columns))
    if not selected:
        raise ValueError("feature_columns cannot be empty")
    if len(selected) != len(set(selected)):
        raise ValueError("feature_columns contains duplicates")
    unknown = set(selected) - set(RAW_FEATURE_COLUMNS)
    if unknown:
        raise ValueError(f"unknown raw feature columns: {sorted(unknown)}")
    return selected


def _split_values(split: str | Sequence[str], rows: int) -> np.ndarray:
    if isinstance(split, str):
        _split_index(split)
        return np.full(rows, split, dtype=object)
    if len(split) != rows:
        raise ValueError("split assignments must match raw rows")
    answer = np.asarray(split, dtype=object)
    for name in set(answer.tolist()):
        _split_index(str(name))
    return answer


def _split_index(split: str) -> int:
    try:
        return SPLIT_NAMES.index(str(split))
    except ValueError as error:
        raise ValueError(f"split must be one of {list(SPLIT_NAMES)}") from error


def _validated_ratios(ratios: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(ratios, Mapping) or set(ratios) != set(SPLIT_NAMES):
        raise ValueError(f"ratios must contain exactly {list(SPLIT_NAMES)}")
    converted = {name: float(ratios[name]) for name in SPLIT_NAMES}
    if any(not math.isfinite(value) or value <= 0 or value >= 1
           for value in converted.values()):
        raise ValueError("each split ratio must be finite and between zero and one")
    if not math.isclose(sum(converted.values()), 1.0, abs_tol=1e-9):
        raise ValueError("split ratios must sum to one")
    return converted


def _minimum_positive_counts(
    value: int | Mapping[str, int],
) -> dict[str, int]:
    if isinstance(value, Mapping):
        unknown = set(value) - set(SPLIT_NAMES)
        missing = set(SPLIT_NAMES) - set(value)
        if unknown or missing:
            raise ValueError(
                "minimum_positives must contain exactly "
                f"{list(SPLIT_NAMES)}"
            )
        converted = {name: int(value[name]) for name in SPLIT_NAMES}
    else:
        converted = {name: int(value) for name in SPLIT_NAMES}
    if any(count < 0 for count in converted.values()):
        raise ValueError("minimum_positives cannot be negative")
    return converted


def _nonnegative_integer_array(values: pd.Series, name: str) -> np.ndarray:
    converted = pd.to_numeric(values, errors="coerce")
    if (converted.isna().any() or (converted < 0).any()
            or not np.all(converted == np.floor(converted))):
        raise ValueError(f"timestamp group {name} must be non-negative integers")
    return converted.to_numpy(dtype=np.int64)


def _validate_counts(
    rows: int,
    positives: int,
    split: str,
    *,
    minimum_positives: int,
    minimum_negatives: int,
) -> None:
    negatives = int(rows) - int(positives)
    if positives < minimum_positives:
        raise ValueError(
            f"{split} has {positives} positive labels; at least "
            f"{minimum_positives} are required"
        )
    if negatives < minimum_negatives:
        raise ValueError(
            f"{split} has {negatives} negative labels; at least "
            f"{minimum_negatives} are required"
        )


def _positive_int(value: Any, name: str) -> int:
    converted = int(value)
    if converted < 1:
        raise ValueError(f"{name} must be at least one")
    return converted


__all__ = [
    "DEFAULT_SPLIT_RATIOS",
    "HASH_BUCKETS",
    "RAW_FEATURE_COLUMNS",
    "RAW_FEATURE_SET_VERSION",
    "RAW_METADATA_COLUMNS",
    "SPLIT_NAMES",
    "RawFeatureChunk",
    "SplitBoundaries",
    "deterministic_negative_mask",
    "deterministic_sample_keys",
    "deterministic_undersample",
    "iter_raw_feature_chunks",
    "plan_chronological_splits",
    "scan_timestamp_groups",
    "timestamp_group_split",
    "transform_raw_chunk",
]
