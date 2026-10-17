"""Build a measured data dictionary for the Sprint 1 transaction pipeline.

The utility profiles a CSV or Parquet dataset in bounded batches and merges the
observed schema with the canonical feature contract.  This makes the output
useful both before the feature store exists (raw-source audit) and afterwards
(measured feature-store null rates).
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow.dataset as pads


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.ml.transform_features import FEATURE_COLUMNS, METADATA_COLUMNS


DEFAULT_FEATURE_ROOT = ROOT / "data" / "processed" / "feature_store"
DEFAULT_RAW = (
    ROOT / "data" / "raw" / "ibm_aml_hi_small" / "HI-Small_Trans.csv"
)
DEFAULT_OUTPUT = ROOT / "docs" / "data_dictionary.csv"

INTEGER_FEATURES = {
    name
    for name in FEATURE_COLUMNS
    if "count" in name
    or name.endswith("_missing")
    or name.endswith("_flag")
    or name.startswith("is_new_")
    or "unique_counterparties" in name
    or "fan_in" in name
    or "fan_out" in name
    or name == "causal_chain_depth"
}

RAW_SPEC = {
    "Timestamp": (
        "Raw transaction time; parsed as UTC-naive source time and normalized to UTC.",
        "identity plus strict datetime parsing",
        False,
        "split_key",
    ),
    "From Bank": (
        "Originating bank identifier.",
        "combined with source account to form source_account",
        False,
        "entity_key",
    ),
    "Account": (
        "Originating account identifier (the first Account column in AMLworld).",
        "combined with From Bank; retained as a string",
        False,
        "entity_key",
    ),
    "To Bank": (
        "Receiving bank identifier.",
        "combined with destination account to form destination_account",
        False,
        "entity_key",
    ),
    "Account.1": (
        "Receiving account identifier (pandas name for the duplicate source header).",
        "combined with To Bank; retained as a string",
        False,
        "entity_key",
    ),
    "Amount Received": (
        "Amount received in the receiving currency.",
        "audit-only raw value; predictive amount uses Amount Paid",
        False,
        "audit",
    ),
    "Receiving Currency": (
        "Currency in which the receiver obtains funds.",
        "audit-only raw category",
        False,
        "audit",
    ),
    "Amount Paid": (
        "Transaction amount in the payment currency.",
        "numeric coercion to canonical amount",
        False,
        "predictor_source",
    ),
    "Payment Currency": (
        "Currency in which the originator pays.",
        "normalized to canonical currency",
        False,
        "predictor_source",
    ),
    "Payment Format": (
        "Payment rail or format.",
        "normalized to canonical transaction_type",
        False,
        "predictor_source",
    ),
    "Is Laundering": (
        "Synthetic laundering ground truth supplied by AMLworld.",
        "renamed label and isolated from feature construction",
        True,
        "target_only",
    ),
}

METADATA_SPEC = {
    "event_id": (
        "Stable transaction identifier used for joins and deterministic queue ties.",
        "deterministic hash or source transaction identifier",
        False,
        "identifier",
    ),
    "transaction_fingerprint": (
        "Stable hash of the source transaction fields used for pattern matching.",
        "deterministic hash over normalized raw transaction fields",
        False,
        "identifier",
    ),
    "source_row_number": (
        "One-based physical source line used as a stable equal-time ordering key.",
        "assigned during bounded-memory source scan",
        False,
        "ordering_key",
    ),
    "dataset": (
        "Dataset adapter that produced the canonical row.",
        "constant assigned by the adapter",
        False,
        "provenance",
    ),
    "time_semantics": (
        "Whether time is calendar UTC or relative simulation time.",
        "assigned by dataset adapter",
        False,
        "provenance",
    ),
    "event_time": (
        "Parsed calendar event time when the source supplies one.",
        "UTC datetime parsing; never inferred for relative-time datasets",
        False,
        "split_key",
    ),
    "relative_time_seconds": (
        "Seconds on the source dataset's relative clock.",
        "source-relative time; no invented calendar origin",
        False,
        "split_key",
    ),
    "source_account": (
        "Canonical originator entity key.",
        "bank and account composite for AMLworld",
        False,
        "entity_key",
    ),
    "destination_account": (
        "Canonical beneficiary entity key when observed.",
        "bank and account composite for AMLworld",
        False,
        "entity_key",
    ),
    "transaction_type": (
        "Canonical payment type or rail.",
        "normalized source category",
        False,
        "context",
    ),
    "currency": (
        "Canonical payment currency.",
        "normalized source category",
        False,
        "context",
    ),
    "device_type": (
        "Optional device class supplied by an identity table, such as IEEE-CIS DeviceType.",
        "optional IEEE-CIS identity-table field joined by TransactionID; unavailable for AMLworld and PaySim",
        False,
        "optional_current_event_context",
    ),
    "device_info": (
        "Optional device descriptor supplied by an identity table, such as IEEE-CIS DeviceInfo.",
        "optional IEEE-CIS identity-table field joined by TransactionID; unavailable for AMLworld and PaySim",
        False,
        "optional_current_event_context",
    ),
    "label_name": (
        "Name and semantics of the target attached to the row.",
        "adapter metadata",
        True,
        "target_metadata_only",
    ),
    "label": (
        "Binary fraud or laundering evaluation target.",
        "numeric coercion of source target; excluded from predictors",
        True,
        "target_only",
    ),
    "case_id": (
        "Synthetic laundering-attempt identifier from the AMLworld pattern file.",
        "ground-truth-only transaction-pattern join",
        True,
        "evaluation_only",
    ),
    "typology": (
        "Synthetic laundering typology from the AMLworld pattern file.",
        "ground-truth-only pattern annotation",
        True,
        "evaluation_and_stratification_only",
    ),
    "case_sequence": (
        "One-based transaction position inside a labeled laundering attempt.",
        "ground-truth-only pattern ordering",
        True,
        "evaluation_only",
    ),
    "pattern_matched": (
        "Whether this transaction occurs in the AMLworld laundering-pattern file.",
        "ground-truth-only fingerprint join",
        True,
        "evaluation_only",
    ),
    "laundering_case_ids": (
        "JSON list of laundering-attempt IDs containing this transaction.",
        "ground-truth-only pattern membership aggregation",
        True,
        "evaluation_only",
    ),
    "laundering_typologies": (
        "JSON list of synthetic typologies associated with this transaction.",
        "ground-truth-only pattern membership aggregation",
        True,
        "evaluation_and_stratification_only",
    ),
    "laundering_case_positions": (
        "JSON case/position records for all pattern memberships.",
        "ground-truth-only pattern membership aggregation",
        True,
        "evaluation_only",
    ),
    "laundering_case_count": (
        "Number of distinct laundering attempts containing the transaction.",
        "ground-truth-only distinct case count",
        True,
        "evaluation_only",
    ),
}


def _feature_spec(name: str) -> tuple[str, str, bool, str]:
    """Return description, transformation, leakage flag, and model role."""
    if name in {"amount", "amount_missing"}:
        description = (
            "Canonical transaction amount."
            if name == "amount"
            else "Indicator that the source amount was absent or non-numeric."
        )
        transform = "numeric coercion from the source amount; missing values are flagged"
    elif name.endswith("_missing"):
        description = f"Missingness indicator for {name.removesuffix('_missing')}."
        transform = "1 when the source value is missing at event time, otherwise 0"
    elif name.startswith("relative_hour_"):
        description = "Cyclical hour-of-day coordinate on the available event clock."
        transform = "sine/cosine encoding calculated from current event time"
    elif name.startswith("relative_day_of_week_"):
        description = "Cyclical day-of-week coordinate on the available event clock."
        transform = "sine/cosine encoding calculated from current event time"
    elif name.startswith("source_txn_count_"):
        window = name.rsplit("_", 1)[-1]
        description = f"Count of source-account transactions in the prior {window}."
        transform = f"strictly past [t-{window}, t) source history; equal-time rows excluded"
    elif name.startswith("source_txn_sum_"):
        window = name.rsplit("_", 1)[-1]
        description = f"Sum of source-account amounts in the prior {window}."
        transform = f"strictly past [t-{window}, t) source history; equal-time rows excluded"
    elif name == "source_amount_mean_30d":
        description = "Mean source-account amount in the preceding 30 days."
        transform = "strictly past [t-30d, t) source history"
    elif name == "amount_to_source_mean_30d_ratio":
        description = "Current amount divided by the source's preceding-30-day mean."
        transform = "current amount / past-only 30d mean; zero when denominator is unavailable"
    elif name == "is_new_counterparty":
        description = "Whether the destination is unseen in the source's retained history."
        transform = "past-only source-to-destination history"
    elif name == "new_device_flag":
        description = "Whether the current device is unseen in the source account's prior history."
        transform = "current device compared only with strictly earlier source-device observations"
    elif name == "source_unique_counterparties_30d":
        description = "Distinct counterparties used by the source during the prior 30 days."
        transform = "strictly past [t-30d, t) source history"
    elif name in {"source_recent_inbound_count", "source_recent_inbound_sum"}:
        description = "Recent inbound activity into the current source account."
        transform = "strictly past inbound graph history within the configured chain window"
    elif name in {"destination_recent_inbound_count", "destination_recent_inbound_sum"}:
        description = "Recent inbound activity into the current destination account."
        transform = "strictly past inbound graph history within the configured chain window"
    elif name == "source_fan_out_24h":
        description = "Distinct destinations reached by the source in the prior 24 hours."
        transform = "strictly past [t-24h, t) graph history"
    elif name == "destination_fan_in_24h":
        description = "Distinct sources paying the destination in the prior 24 hours."
        transform = "strictly past [t-24h, t) graph history"
    elif name == "rapid_pass_through_ratio":
        description = "Current outbound amount relative to recent inbound funds at the source."
        transform = "current amount / strictly prior inbound sum within chain window"
    elif name == "causal_chain_depth":
        description = "Past-only estimated transfer-chain depth ending at this event."
        transform = "bounded recurrence over edges whose timestamps are strictly earlier"
    elif name.startswith("paysim_prior_transfer_in_"):
        description = "Prior PaySim TRANSFER activity into the current source account."
        transform = "strictly past PaySim type-filtered 24h history"
    elif name.startswith("paysim_transfer_to_cashout_"):
        description = "Past-only PaySim transfer-to-cash-out behavior indicator or ratio."
        transform = "current CASH_OUT compared with strictly prior inbound TRANSFER history"
    else:
        description = name.replace("_", " ").capitalize() + "."
        transform = "deterministic, label-blind transformation available at event time"
    return description, transform, False, "predictor"


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
        source = pd.read_csv(path, chunksize=batch_size, low_memory=False)
        iterator: Iterator[pd.DataFrame] = iter(source)
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


def profile(path: Path, batch_size: int, max_rows: int | None) -> tuple[int, dict[str, dict[str, object]]]:
    total_rows = 0
    present: dict[str, int] = defaultdict(int)
    missing: dict[str, int] = defaultdict(int)
    dtypes: dict[str, set[str]] = defaultdict(set)
    order: list[str] = []

    for frame in iter_frames(path, batch_size, max_rows):
        rows = len(frame)
        total_rows += rows
        for column in frame.columns:
            name = str(column)
            if name not in present:
                order.append(name)
            present[name] += rows
            missing[name] += int(frame[column].isna().sum())
            dtypes[name].add(str(frame[column].dtype))

    if total_rows == 0:
        raise ValueError(f"input contains no rows: {path}")
    result: dict[str, dict[str, object]] = {}
    for name in order:
        # If a column is absent from a fragment, those rows are structurally null.
        null_count = missing[name] + total_rows - present[name]
        dtype = sorted(dtypes[name])
        result[name] = {
            "dtype": dtype[0] if len(dtype) == 1 else "mixed[" + ",".join(dtype) + "]",
            "null_rate": null_count / total_rows,
        }
    return total_rows, result


def _expected_dtype(name: str) -> str:
    if name in INTEGER_FEATURES:
        return "int64"
    if name in FEATURE_COLUMNS:
        return "float64"
    return {
        "event_time": "datetime64[ns, UTC]",
        "relative_time_seconds": "float64",
        "label": "int64",
        "case_sequence": "Int64",
    }.get(name, "string")


def _spec(name: str, input_name: str) -> tuple[str, str, bool, str, str]:
    if name in RAW_SPEC:
        description, transform, risk, role = RAW_SPEC[name]
        source = input_name
    elif name in METADATA_SPEC:
        description, transform, risk, role = METADATA_SPEC[name]
        if name in {"device_type", "device_info"}:
            source = "IEEE-CIS identity table (optional; absent from AMLworld and PaySim)"
        elif name in {
            "case_id", "typology", "case_sequence", "pattern_matched",
            "laundering_case_ids", "laundering_typologies",
            "laundering_case_positions", "laundering_case_count",
        }:
            source = "HI-Small_Patterns.txt (evaluation only)"
        elif name in {"transaction_fingerprint", "source_row_number"}:
            source = "raw transaction file"
        else:
            source = "canonical transaction record"
    elif name in FEATURE_COLUMNS:
        description, transform, risk, role = _feature_spec(name)
        source = "current transaction + strictly prior transaction history"
    else:
        description = name.replace("_", " ").capitalize() + "."
        transform = "observed column; see producing pipeline manifest"
        normalized = "".join(char for char in name.casefold() if char.isalnum())
        risk = normalized in {
            "isfraud", "islaundering", "label", "target", "groundtruth"
        }
        role = "target_only" if risk else "observed"
        source = input_name
    return description, transform, risk, role, source


def build_rows(
    observed: dict[str, dict[str, object]], input_path: Path, include_expected: bool
) -> list[dict[str, object]]:
    names = list(observed)
    if include_expected:
        for name in [
            *METADATA_COLUMNS,
            "transaction_fingerprint",
            "source_row_number",
            "pattern_matched",
            "laundering_case_ids",
            "laundering_typologies",
            "laundering_case_positions",
            "laundering_case_count",
            *FEATURE_COLUMNS,
        ]:
            if name not in names:
                names.append(name)

    rows: list[dict[str, object]] = []
    for name in names:
        description, transformation, risk, role, source = _spec(name, input_path.name)
        values = observed.get(name)
        rows.append(
            {
                "feature_name": name,
                "dtype": values["dtype"] if values else _expected_dtype(name),
                "source_table": source,
                "null_rate": (
                    f"{float(values['null_rate']):.8f}" if values else ""
                ),
                "transformation_applied": transformation,
                "leakage_risk_flag": str(bool(risk)).lower(),
                "model_role": role,
                "description": description,
                "observed_in_input": str(values is not None).lower(),
            }
        )
    return rows


def write_dictionary(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        help="CSV/Parquet file or Hive-partitioned Parquet directory. Defaults to the feature store, then raw AMLworld.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--batch-size", type=int, default=200_000)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument(
        "--observed-only",
        action="store_true",
        help="Do not append canonical columns that have not yet been produced.",
    )
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.max_rows is not None and args.max_rows < 1:
        parser.error("--max-rows must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = resolve_input(args.input)
    total_rows, observed = profile(input_path, args.batch_size, args.max_rows)
    rows = build_rows(observed, input_path, not args.observed_only)
    output = Path(args.output).expanduser().resolve()
    write_dictionary(rows, output)
    print(f"Wrote {len(rows)} dictionary entries to {output}")
    print(f"Profiled {total_rows:,} rows from {input_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
