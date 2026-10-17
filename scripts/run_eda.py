"""Run bounded-memory exploratory analysis for raw or processed transaction data.

Outputs are machine-readable first (JSON/CSV), with two compact publication-
ready PNGs.  CSV input is read in chunks and Parquet input in Arrow record
batches; no operation requires the full transaction table in memory.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "capacity_aware_aml_matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as pads


DEFAULT_FEATURE_ROOT = ROOT / "data" / "processed" / "feature_store"
DEFAULT_RAW = (
    ROOT / "data" / "raw" / "ibm_aml_hi_small" / "HI-Small_Trans.csv"
)
DEFAULT_PATTERNS = (
    ROOT / "data" / "raw" / "ibm_aml_hi_small" / "HI-Small_Patterns.txt"
)
DEFAULT_OUTPUT = ROOT / "reports" / "eda"


def _normalise(value: object) -> str:
    return "".join(char for char in str(value).casefold() if char.isalnum())


def _find(columns: list[str], *candidates: str) -> str | None:
    lookup = {_normalise(column): column for column in columns}
    for candidate in candidates:
        if _normalise(candidate) in lookup:
            return lookup[_normalise(candidate)]
    return None


def _contains_parquet(path: Path) -> bool:
    return path.is_dir() and any(path.rglob("*.parquet"))


def resolve_input(value: str | None) -> Path:
    if value:
        path = Path(value).expanduser().resolve()
    elif _contains_parquet(DEFAULT_FEATURE_ROOT):
        path = DEFAULT_FEATURE_ROOT
    else:
        path = DEFAULT_RAW
    if not path.exists():
        raise FileNotFoundError(
            f"input not found: {path}. Acquire data or pass --input explicitly."
        )
    if path.is_dir() and not _contains_parquet(path):
        raise FileNotFoundError(f"no Parquet files found under input directory: {path}")
    return path


def iter_frames(path: Path, batch_size: int, max_rows: int | None) -> Iterator[pd.DataFrame]:
    emitted = 0
    if path.is_file() and path.suffix.casefold() in {".csv", ".txt"}:
        reader = pd.read_csv(path, chunksize=batch_size, low_memory=False)
        iterator: Iterator[pd.DataFrame] = iter(reader)
    elif path.is_file() and path.suffix.casefold() in {".parquet", ".pq"}:
        dataset = pads.dataset(str(path), format="parquet")
        iterator = (batch.to_pandas() for batch in dataset.to_batches(batch_size=batch_size))
    elif path.is_dir():
        dataset = pads.dataset(
            str(path), format="parquet", partitioning="hive", exclude_invalid_files=True
        )
        iterator = (batch.to_pandas() for batch in dataset.to_batches(batch_size=batch_size))
    else:
        raise ValueError(f"unsupported input type: {path}")

    for frame in iterator:
        if max_rows is not None:
            remaining = max_rows - emitted
            if remaining <= 0:
                break
            frame = frame.iloc[:remaining]
        if frame.empty:
            continue
        emitted += len(frame)
        yield frame


@dataclass
class RunningNumeric:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def update(self, series: pd.Series) -> None:
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            return
        new_count = int(len(values))
        new_mean = float(values.mean())
        new_m2 = float(np.square(values - new_mean).sum())
        if self.count == 0:
            self.count, self.mean, self.m2 = new_count, new_mean, new_m2
        else:
            delta = new_mean - self.mean
            combined = self.count + new_count
            self.m2 += new_m2 + delta * delta * self.count * new_count / combined
            self.mean += delta * new_count / combined
            self.count = combined
        self.minimum = min(self.minimum, float(values.min()))
        self.maximum = max(self.maximum, float(values.max()))

    def row(self, name: str) -> dict[str, object]:
        return {
            "column": name,
            "count": self.count,
            "mean": self.mean if self.count else "",
            "std": math.sqrt(self.m2 / (self.count - 1)) if self.count > 1 else "",
            "min": self.minimum if self.count else "",
            "max": self.maximum if self.count else "",
        }


class PrioritySample:
    """Uniform fixed-size sample using independent random priorities."""

    def __init__(self, capacity: int, seed: int) -> None:
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.keys = np.empty(0, dtype=float)
        self.values = np.empty(0, dtype=float)

    def update(self, series: pd.Series) -> None:
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            return
        keys = self.rng.random(len(values))
        all_keys = np.concatenate([self.keys, keys])
        all_values = np.concatenate([self.values, values])
        if len(all_keys) > self.capacity:
            indices = np.argpartition(all_keys, self.capacity - 1)[: self.capacity]
            all_keys, all_values = all_keys[indices], all_values[indices]
        self.keys, self.values = all_keys, all_values


def parse_patterns(path: Path | None) -> tuple[dict[str, object], list[dict[str, object]]]:
    if path is None or not path.exists():
        return {"available": False}, []
    begin = re.compile(r"^BEGIN LAUNDERING ATTEMPT -\s*([^:]+?)(?::\s*(.*))?$", re.I)
    end = re.compile(r"^END LAUNDERING ATTEMPT -\s*(.+?)\s*$", re.I)
    cases: Counter[str] = Counter()
    transactions: Counter[str] = Counter()
    sizes: defaultdict[str, list[int]] = defaultdict(list)
    current: str | None = None
    current_size = 0
    malformed = 0
    total_lines = 0
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            total_lines += 1
            start = begin.match(line)
            if start:
                if current is not None:
                    malformed += 1
                current = re.sub(r"\s+", "-", start.group(1).strip().upper())
                current_size = 0
                cases[current] += 1
                continue
            if end.match(line):
                if current is None:
                    malformed += 1
                else:
                    sizes[current].append(current_size)
                current, current_size = None, 0
                continue
            if current is None:
                malformed += 1
            else:
                current_size += 1
                transactions[current] += 1
    if current is not None:
        malformed += 1
        sizes[current].append(current_size)
    rows = []
    for typology in sorted(cases):
        case_sizes = sizes[typology]
        rows.append(
            {
                "typology": typology,
                "cases": int(cases[typology]),
                "transactions": int(transactions[typology]),
                "mean_transactions_per_case": (
                    float(np.mean(case_sizes)) if case_sizes else 0.0
                ),
                "min_transactions_per_case": min(case_sizes) if case_sizes else 0,
                "max_transactions_per_case": max(case_sizes) if case_sizes else 0,
            }
        )
    return (
        {
            "available": True,
            "path": str(path),
            "nonblank_lines": total_lines,
            "case_count": int(sum(cases.values())),
            "transaction_annotations": int(sum(transactions.values())),
            "typology_count": len(cases),
            "malformed_or_orphan_lines": malformed,
        },
        rows,
    )


def _write_csv(rows: list[dict[str, object]], path: Path, columns: list[str]) -> None:
    frame = pd.DataFrame(rows, columns=columns)
    frame.to_csv(path, index=False)


def _manifest_for(path: Path) -> dict[str, object] | None:
    candidates = [path.parent / "manifest.json", path / "manifest.json"]
    for candidate in candidates:
        if candidate.exists():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return {"path": str(candidate), "read_error": True}
    return None


def run_eda(
    input_path: Path,
    output_dir: Path,
    batch_size: int,
    max_rows: int | None,
    pattern_path: Path | None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    batch_count = 0
    nulls: Counter[str] = Counter()
    present: Counter[str] = Counter()
    dtype_map: defaultdict[str, set[str]] = defaultdict(set)
    column_order: list[str] = []
    numeric: dict[str, RunningNumeric] = {}
    categories: defaultdict[str, Counter[str]] = defaultdict(Counter)
    source_hashes: set[int] = set()
    destination_hashes: set[int] = set()
    label_counts: Counter[str] = Counter()
    daily_count: Counter[str] = Counter()
    daily_amount: Counter[str] = Counter()
    amount_sample = PrioritySample(50_000, seed=20261007)

    fields: dict[str, str | None] | None = None
    invalid_time = invalid_amount = negative_amount = zero_amount = 0
    invalid_label = 0
    chronology_regressions = 0
    previous_timestamp_ns: int | None = None
    min_timestamp: pd.Timestamp | None = None
    max_timestamp: pd.Timestamp | None = None

    for frame in iter_frames(input_path, batch_size, max_rows):
        batch_count += 1
        rows = len(frame)
        total_rows += rows
        columns = [str(column) for column in frame.columns]
        if fields is None:
            fields = {
                "timestamp": _find(columns, "event_time", "Timestamp", "datetime", "time"),
                "amount": _find(columns, "amount", "Amount Paid", "TransactionAmt"),
                "label": _find(columns, "label", "Is Laundering", "isFraud", "isFraud"),
                "type": _find(columns, "transaction_type", "Payment Format", "type"),
                "currency": _find(columns, "currency", "Payment Currency"),
                "receiving_currency": _find(columns, "Receiving Currency"),
                "source_bank": _find(columns, "From Bank", "source_bank"),
                "destination_bank": _find(columns, "To Bank", "destination_bank"),
                "source": _find(columns, "source_account", "Account", "nameOrig"),
                "destination": _find(columns, "destination_account", "Account.1", "nameDest"),
            }

        for column in frame.columns:
            name = str(column)
            if name not in present:
                column_order.append(name)
            present[name] += rows
            nulls[name] += int(frame[column].isna().sum())
            dtype_map[name].add(str(frame[column].dtype))
            if pd.api.types.is_numeric_dtype(frame[column]):
                numeric.setdefault(name, RunningNumeric()).update(frame[column])

        assert fields is not None
        low_cardinality = [
            fields["label"], fields["type"], fields["currency"],
            fields["receiving_currency"], fields["source_bank"],
            fields["destination_bank"],
        ]
        for column in dict.fromkeys(value for value in low_cardinality if value):
            values = frame[column].astype("string").fillna("<MISSING>")
            categories[column].update(map(str, values))

        label_column = fields["label"]
        if label_column:
            labels = pd.to_numeric(frame[label_column], errors="coerce")
            invalid_label += int((labels.notna() & ~labels.isin([0, 1])).sum())
            label_counts.update(
                "missing" if pd.isna(value) else str(int(value))
                if float(value).is_integer()
                else str(value)
                for value in labels
            )

        amount_column = fields["amount"]
        amounts: pd.Series | None = None
        if amount_column:
            amounts = pd.to_numeric(frame[amount_column], errors="coerce")
            invalid_amount += int(amounts.isna().sum() - frame[amount_column].isna().sum())
            negative_amount += int(amounts.lt(0).sum())
            zero_amount += int(amounts.eq(0).sum())
            # Numeric source columns were already accumulated in the generic
            # loop. String-typed amount columns still need numeric coercion.
            if not pd.api.types.is_numeric_dtype(frame[amount_column]):
                numeric.setdefault(amount_column, RunningNumeric()).update(amounts)
            amount_sample.update(amounts)

        time_column = fields["timestamp"]
        if time_column:
            timestamps = pd.to_datetime(frame[time_column], errors="coerce", utc=True)
            invalid_time += int(timestamps.isna().sum() - frame[time_column].isna().sum())
            valid = timestamps.dropna()
            if len(valid):
                current_min, current_max = valid.min(), valid.max()
                min_timestamp = current_min if min_timestamp is None else min(min_timestamp, current_min)
                max_timestamp = current_max if max_timestamp is None else max(max_timestamp, current_max)
                values_ns = valid.astype("int64").to_numpy()
                chronology_regressions += int(np.count_nonzero(np.diff(values_ns) < 0))
                if previous_timestamp_ns is not None and int(values_ns[0]) < previous_timestamp_ns:
                    chronology_regressions += 1
                previous_timestamp_ns = int(values_ns[-1])
                day = timestamps.dt.strftime("%Y-%m-%d").fillna("<MISSING>")
                day_counts = day.value_counts()
                for key, value in day_counts.items():
                    daily_count[str(key)] += int(value)
                if amounts is not None:
                    grouped = pd.DataFrame({"day": day, "amount": amounts}).groupby(
                        "day", dropna=False
                    )["amount"].sum(min_count=1)
                    for key, value in grouped.items():
                        if pd.notna(value):
                            daily_amount[str(key)] += float(value)

        for key, target in (("source", source_hashes), ("destination", destination_hashes)):
            column = fields[key]
            if column:
                values = frame[column].dropna().astype("string")
                hashes = pd.util.hash_pandas_object(values, index=False).to_numpy(dtype="uint64")
                target.update(map(int, hashes))

    if total_rows == 0 or fields is None:
        raise ValueError(f"input contains no rows: {input_path}")

    column_rows = []
    for name in column_order:
        structural_null = total_rows - present[name]
        null_count = nulls[name] + structural_null
        dtype_values = sorted(dtype_map[name])
        column_rows.append(
            {
                "column": name,
                "dtype": dtype_values[0] if len(dtype_values) == 1 else "mixed[" + ",".join(dtype_values) + "]",
                "null_count": int(null_count),
                "null_rate": null_count / total_rows,
            }
        )
    _write_csv(column_rows, output_dir / "column_profile.csv", list(column_rows[0]))

    numeric_rows = [accumulator.row(name) for name, accumulator in sorted(numeric.items())]
    _write_csv(
        numeric_rows,
        output_dir / "numeric_profile.csv",
        ["column", "count", "mean", "std", "min", "max"],
    )

    category_rows = []
    for column, counts in sorted(categories.items()):
        denominator = sum(counts.values())
        for value, count in counts.most_common(20):
            category_rows.append(
                {
                    "column": column,
                    "value": value,
                    "count": int(count),
                    "share": count / denominator if denominator else 0.0,
                }
            )
    _write_csv(
        category_rows,
        output_dir / "categorical_top_values.csv",
        ["column", "value", "count", "share"],
    )

    daily_rows = [
        {"date": day, "transactions": daily_count[day], "amount": daily_amount[day]}
        for day in sorted(daily_count)
    ]
    _write_csv(
        daily_rows,
        output_dir / "daily_activity.csv",
        ["date", "transactions", "amount"],
    )

    pattern_summary, pattern_rows = parse_patterns(pattern_path)
    _write_csv(
        pattern_rows,
        output_dir / "typology_profile.csv",
        [
            "typology", "cases", "transactions", "mean_transactions_per_case",
            "min_transactions_per_case", "max_transactions_per_case",
        ],
    )

    label_total = sum(label_counts.values())
    positives = label_counts.get("1", 0)
    source_missing = (
        int(nulls[fields["source"]]) if fields["source"] is not None else total_rows
    )
    destination_missing = (
        int(nulls[fields["destination"]]) if fields["destination"] is not None else total_rows
    )
    quality_rows = [
        {
            "check": "timestamp_parse_failures",
            "value": invalid_time,
            "status": "pass" if invalid_time == 0 else "warn",
        },
        {
            "check": "input_order_timestamp_regressions",
            "value": chronology_regressions,
            "status": "pass" if chronology_regressions == 0 else "warn",
        },
        {
            "check": "amount_parse_failures",
            "value": invalid_amount,
            "status": "pass" if invalid_amount == 0 else "warn",
        },
        {
            "check": "negative_amounts",
            "value": negative_amount,
            "status": "pass" if negative_amount == 0 else "warn",
        },
        {
            "check": "non_binary_labels",
            "value": invalid_label,
            "status": "pass" if invalid_label == 0 else "fail",
        },
        {
            "check": "missing_source_entities",
            "value": source_missing,
            "status": "pass" if source_missing == 0 else "warn",
        },
        {
            "check": "missing_destination_entities",
            "value": destination_missing,
            "status": "pass" if destination_missing == 0 else "warn",
        },
    ]
    _write_csv(quality_rows, output_dir / "quality_checks.csv", ["check", "value", "status"])

    quantiles: dict[str, float] = {}
    if len(amount_sample.values):
        for probability in (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0):
            quantiles[f"p{int(probability * 100):02d}"] = float(
                np.quantile(amount_sample.values, probability)
            )
    summary: dict[str, object] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "input": str(input_path),
        "input_manifest": _manifest_for(input_path),
        "profile_scope": {
            "rows": total_rows,
            "batches": batch_count,
            "batch_size": batch_size,
            "max_rows": max_rows,
            "is_bounded_sample": max_rows is not None and total_rows >= max_rows,
        },
        "resolved_fields": fields,
        "time": {
            "minimum_utc": min_timestamp.isoformat() if min_timestamp is not None else None,
            "maximum_utc": max_timestamp.isoformat() if max_timestamp is not None else None,
            "input_order_regressions": chronology_regressions,
        },
        "class_balance": {
            "counts": dict(sorted(label_counts.items())),
            "positive_count": positives,
            "positive_rate": positives / label_total if label_total else None,
        },
        "amount": {
            "parse_failures": invalid_amount,
            "negative_count": negative_amount,
            "zero_count": zero_amount,
            "sample_size_for_quantiles": int(len(amount_sample.values)),
            "sample_quantiles": quantiles,
        },
        "entities": {
            "source_distinct_hash64": len(source_hashes),
            "destination_distinct_hash64": len(destination_hashes),
            "note": "Counts use distinct deterministic 64-bit pandas hashes to bound memory.",
        },
        "pattern_ground_truth": pattern_summary,
        "quality": {row["check"]: row["value"] for row in quality_rows},
        "interpretation_guardrails": [
            "AMLworld is synthetic; prevalence is not a production base-rate estimate.",
            "Pattern typology and label fields are evaluation-only and must not enter predictors.",
            "Timestamp regressions in the raw file require chronological sorting before causal features.",
            "A max-row run is a development sample, not the final descriptive analysis.",
        ],
    }
    temporary = output_dir / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(output_dir / "summary.json")

    _make_plots(output_dir, label_counts, daily_rows, amount_sample.values)
    return summary


def _make_plots(
    output_dir: Path,
    label_counts: Counter[str],
    daily_rows: list[dict[str, object]],
    amount_values: np.ndarray,
) -> None:
    if label_counts:
        labels = sorted(label_counts, key=lambda value: (value == "missing", value))
        counts = [label_counts[label] for label in labels]
        fig, axis = plt.subplots(figsize=(6.4, 4.0))
        axis.bar(labels, counts, color=["#4C78A8", "#E45756", "#9D9D9D"][: len(labels)])
        axis.set(title="Transaction label distribution", xlabel="Label", ylabel="Transactions")
        axis.ticklabel_format(style="plain", axis="y")
        for index, count in enumerate(counts):
            axis.text(index, count, f"{count:,}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / "class_balance.png", dpi=180)
        plt.close(fig)

    plot_rows = [row for row in daily_rows if row["date"] != "<MISSING>"]
    if plot_rows:
        dates = pd.to_datetime([row["date"] for row in plot_rows])
        counts = [int(row["transactions"]) for row in plot_rows]
        fig, axis = plt.subplots(figsize=(9.0, 4.2))
        axis.plot(dates, counts, marker="o", linewidth=1.8, color="#4C78A8")
        axis.set(title="Transactions by event date", xlabel="Event date", ylabel="Transactions")
        axis.ticklabel_format(style="plain", axis="y")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(output_dir / "daily_volume.png", dpi=180)
        plt.close(fig)

    positive = amount_values[np.isfinite(amount_values) & (amount_values > 0)]
    if len(positive):
        fig, axis = plt.subplots(figsize=(7.2, 4.2))
        axis.hist(np.log10(positive), bins=50, color="#72B7B2", edgecolor="white")
        axis.set(
            title="Sampled positive transaction amounts",
            xlabel="log10(amount)",
            ylabel="Sampled transactions",
        )
        fig.tight_layout()
        fig.savefig(output_dir / "amount_distribution.png", dpi=180)
        plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        help="CSV/Parquet file or Hive-partitioned directory. Defaults to data/processed/feature_store, then raw AMLworld.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--patterns", default=str(DEFAULT_PATTERNS))
    parser.add_argument("--batch-size", type=int, default=200_000)
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.max_rows is not None and args.max_rows < 1:
        parser.error("--max-rows must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = resolve_input(args.input)
    pattern_path = Path(args.patterns).expanduser().resolve() if args.patterns else None
    output_dir = Path(args.output_dir).expanduser().resolve()
    summary = run_eda(input_path, output_dir, args.batch_size, args.max_rows, pattern_path)
    print(f"Wrote EDA artifacts to {output_dir}")
    print(f"Profiled {int(summary['profile_scope']['rows']):,} rows from {input_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
