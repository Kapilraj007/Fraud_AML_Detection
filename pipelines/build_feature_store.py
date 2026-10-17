"""Build the leakage-safe, time-partitioned AMLworld feature store.

The AMLworld transaction CSV is not globally chronological.  This pipeline
therefore performs a disk-backed date bucketing pass, stably orders every
bucket by timestamp and original source row, and only then invokes the shared
stateful feature transformer.  ``--max-rows`` limits the earliest rows after
that ordering; it never means "first physical rows in the CSV."
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import pandas as pd

from backend.app.ml.transform_features import FEATURE_COLUMNS, transform_features
from pipelines.parse_amlworld_patterns import (
    AMLDATA_COLUMNS,
    fingerprint_frame,
    parse_pattern_file,
)


DEFAULT_INPUT = Path("data/raw/ibm_aml_hi_small/HI-Small_Trans.csv")
DEFAULT_PATTERNS = Path("data/raw/ibm_aml_hi_small/HI-Small_Patterns.txt")
DEFAULT_OUTPUT = Path("data/processed/feature_store")
DEFAULT_REGISTRY = Path("config/datasets.json")
DATASET_ID = "ibm_aml_hi_small"


def file_sha256(path: str | Path, block_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def build_feature_store(
    input_path: str | Path = DEFAULT_INPUT,
    patterns_path: str | Path = DEFAULT_PATTERNS,
    output_path: str | Path = DEFAULT_OUTPUT,
    *,
    chunk_size: int = 100_000,
    max_rows: int | None = None,
    verify_hash: bool = True,
    registry_path: str | Path = DEFAULT_REGISTRY,
) -> dict[str, Any]:
    """Build features and return the emitted manifest.

    The destination must not already exist.  This makes failed or accidental
    reruns non-destructive; work is staged beside the destination and renamed
    only after all validation succeeds.
    """
    source = Path(input_path).resolve()
    patterns = Path(patterns_path).resolve()
    destination = Path(output_path).resolve()
    registry = Path(registry_path).resolve()
    _validate_arguments(source, patterns, destination, chunk_size, max_rows)
    destination.parent.mkdir(parents=True, exist_ok=True)

    source_hashes = _source_hashes(source, patterns, registry, verify_hash)
    memberships = parse_pattern_file(patterns)
    pattern_lookup = _pattern_lookup(memberships)
    expected_pattern_fingerprints = set(pattern_lookup)

    with tempfile.TemporaryDirectory(
        prefix=".feature-store-build-", dir=destination.parent
    ) as temporary:
        temporary_root = Path(temporary)
        spool_root = temporary_root / "chronological-spool"
        staged_output = temporary_root / "feature-store"
        spool_root.mkdir()
        staged_output.mkdir()

        source_rows, source_min, source_max = _bucket_source(
            source, spool_root, chunk_size, max_rows=max_rows
        )
        requested_rows = source_rows if max_rows is None else min(max_rows, source_rows)
        result = _transform_buckets(
            spool_root=spool_root,
            output_root=staged_output,
            lookup=pattern_lookup,
            chunk_size=chunk_size,
            max_rows=requested_rows,
        )

        if max_rows is None:
            unmatched = expected_pattern_fingerprints - result["matched_fingerprints"]
            if unmatched:
                examples = sorted(unmatched)[:3]
                raise ValueError(
                    f"{len(unmatched)} pattern transactions did not match the raw CSV; "
                    f"examples: {examples}"
                )

        manifest = {
            "schema_version": 1,
            "dataset_id": DATASET_ID,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": {
                "transaction_file": str(source),
                "pattern_file": str(patterns),
                "sha256": source_hashes,
                "physical_rows_scanned": source_rows,
                "minimum_event_time": _iso(source_min),
                "maximum_event_time": _iso(source_max),
            },
            "ordering": {
                "keys": ["event_time", "source_row_number"],
                "stable": True,
                "external_date_bucketing": True,
                "max_rows_semantics": "earliest rows after global chronological order",
            },
            "output": {
                "rows": result["rows"],
                "partitions": result["partitions"],
                "parquet_files": result["parquet_files"],
                "minimum_event_time": _iso(result["minimum_event_time"]),
                "maximum_event_time": _iso(result["maximum_event_time"]),
            },
            "patterns": {
                "case_count": int(memberships["case_id"].nunique()),
                "membership_count": int(len(memberships)),
                "unique_pattern_transactions": len(expected_pattern_fingerprints),
                "matched_unique_transactions": len(result["matched_fingerprints"]),
                "complete_source_match_required": max_rows is None,
            },
            "features": {
                "transform": "backend.app.ml.transform_features.transform_features",
                "causal": True,
                "feature_columns": list(FEATURE_COLUMNS),
                "ground_truth_metadata_columns": [
                    "label",
                    "laundering_case_ids",
                    "laundering_typologies",
                    "laundering_case_positions",
                    "pattern_matched",
                ],
            },
            "parameters": {"chunk_size": chunk_size, "max_rows": max_rows},
        }
        _write_json(staged_output / "manifest.json", manifest)
        staged_output.replace(destination)
    return manifest


def _validate_arguments(
    source: Path,
    patterns: Path,
    destination: Path,
    chunk_size: int,
    max_rows: int | None,
) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    if not patterns.is_file():
        raise FileNotFoundError(patterns)
    if destination.exists():
        raise FileExistsError(
            f"destination already exists: {destination}; choose a new path or remove it explicitly"
        )
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least one")
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be at least one")


def _source_hashes(
    source: Path, patterns: Path, registry_path: Path, verify: bool
) -> dict[str, str]:
    observed = {source.name: file_sha256(source), patterns.name: file_sha256(patterns)}
    if not verify:
        return observed
    if not registry_path.is_file():
        raise FileNotFoundError(f"dataset registry not found: {registry_path}")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    try:
        registered = {
            item["name"]: item.get("sha256")
            for item in registry["datasets"][DATASET_ID]["files"]
        }
    except (KeyError, TypeError) as exc:
        raise ValueError(f"invalid dataset registry: {registry_path}") from exc
    for name, digest in observed.items():
        expected = registered.get(name)
        if not expected:
            raise ValueError(f"no trusted SHA-256 registered for {name}")
        if digest.casefold() != str(expected).casefold():
            raise ValueError(
                f"SHA-256 mismatch for {name}: expected {expected}, observed {digest}"
            )
    return observed


def _bucket_source(
    source: Path,
    spool_root: Path,
    chunk_size: int,
    *,
    max_rows: int | None,
) -> tuple[int, pd.Timestamp, pd.Timestamp]:
    """Read every row and retain globally earliest rows when a limit is set."""
    file_numbers: defaultdict[str, int] = defaultdict(int)
    rows_seen = 0
    minimum: pd.Timestamp | None = None
    maximum: pd.Timestamp | None = None
    earliest: pd.DataFrame | None = None
    reader = pd.read_csv(
        source,
        dtype=str,
        keep_default_na=False,
        chunksize=chunk_size,
        encoding="utf-8-sig",
    )
    for raw in reader:
        missing = [column for column in AMLDATA_COLUMNS if column not in raw.columns]
        if missing:
            raise ValueError(f"transaction CSV is missing columns: {missing}")
        count = len(raw)
        raw["source_row_number"] = range(rows_seen + 2, rows_seen + count + 2)
        event_time = pd.to_datetime(
            raw["Timestamp"], format="%Y/%m/%d %H:%M", errors="coerce", utc=True
        )
        if event_time.isna().any():
            source_line = int(raw.loc[event_time.isna(), "source_row_number"].iloc[0])
            value = raw.loc[event_time.isna(), "Timestamp"].iloc[0]
            raise ValueError(f"invalid Timestamp {value!r} at source line {source_line}")
        raw["_event_ts"] = event_time
        chunk_min, chunk_max = event_time.min(), event_time.max()
        minimum = chunk_min if minimum is None else min(minimum, chunk_min)
        maximum = chunk_max if maximum is None else max(maximum, chunk_max)
        if max_rows is None:
            _spool_frame(raw, spool_root, file_numbers)
        else:
            candidate = raw.nsmallest(
                max_rows, ["_event_ts", "source_row_number"], keep="first"
            )
            earliest = (
                candidate
                if earliest is None
                else pd.concat([earliest, candidate], ignore_index=True).nsmallest(
                    max_rows, ["_event_ts", "source_row_number"], keep="first"
                )
            )
        rows_seen += count
    if rows_seen == 0 or minimum is None or maximum is None:
        raise ValueError("transaction CSV contains no data rows")
    if max_rows is not None:
        assert earliest is not None
        _spool_frame(earliest, spool_root, file_numbers)
    return rows_seen, minimum, maximum


def _spool_frame(
    frame: pd.DataFrame,
    spool_root: Path,
    file_numbers: defaultdict[str, int],
) -> None:
    frame = frame.copy()
    frame["transaction_fingerprint"] = fingerprint_frame(frame)
    frame["transaction_id"] = [
        f"{fingerprint}-r{int(line):09d}"
        for fingerprint, line in zip(
            frame["transaction_fingerprint"], frame["source_row_number"]
        )
    ]
    frame["_event_date"] = frame["_event_ts"].dt.strftime("%Y-%m-%d")
    for event_date, group in frame.groupby("_event_date", sort=False):
        partition = spool_root / f"event_date={event_date}"
        partition.mkdir(parents=True, exist_ok=True)
        number = file_numbers[event_date]
        file_numbers[event_date] += 1
        payload = group.drop(columns=["_event_date"])
        payload.to_parquet(
            partition / f"raw-{number:06d}.parquet",
            index=False,
            compression="zstd",
        )


def _pattern_lookup(memberships: pd.DataFrame) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for fingerprint, group in memberships.groupby("transaction_fingerprint", sort=False):
        ordered = group.sort_values(
            ["case_id", "case_position", "pattern_line_number"], kind="mergesort"
        )
        lookup[str(fingerprint)] = {
            "laundering_case_ids": json.dumps(
                list(dict.fromkeys(ordered["case_id"].astype(str))), separators=(",", ":")
            ),
            "laundering_typologies": json.dumps(
                list(dict.fromkeys(ordered["typology"].astype(str))), separators=(",", ":")
            ),
            "laundering_case_positions": json.dumps(
                [
                    {"case_id": str(row.case_id), "position": int(row.case_position)}
                    for row in ordered.itertuples()
                ],
                separators=(",", ":"),
            ),
            "laundering_case_count": int(ordered["case_id"].nunique()),
        }
    return lookup


def _transform_buckets(
    spool_root: Path,
    output_root: Path,
    lookup: Mapping[str, Mapping[str, Any]],
    chunk_size: int,
    max_rows: int,
) -> dict[str, Any]:
    state: dict[str, Any] = {}
    written = 0
    files_written = 0
    partitions = 0
    output_part = 0
    observed_matches: set[str] = set()
    minimum: pd.Timestamp | None = None
    maximum: pd.Timestamp | None = None

    for day_directory in sorted(spool_root.glob("event_date=*")):
        if written >= max_rows:
            break
        raw_files = sorted(day_directory.glob("raw-*.parquet"))
        daily = pd.concat((pd.read_parquet(path) for path in raw_files), ignore_index=True)
        daily = daily.sort_values(
            ["_event_ts", "source_row_number"], kind="mergesort"
        ).reset_index(drop=True)
        remaining = max_rows - written
        if len(daily) > remaining:
            daily = daily.iloc[:remaining].copy()
        event_date = day_directory.name.split("=", 1)[1]
        output_partition = output_root / day_directory.name
        output_partition.mkdir(parents=True, exist_ok=True)
        partitions += 1

        for start in range(0, len(daily), chunk_size):
            raw_chunk = daily.iloc[start : start + chunk_size].reset_index(drop=True)
            featured = transform_features(raw_chunk, "amlworld", state=state)
            fingerprints = raw_chunk["transaction_fingerprint"].astype(str)
            metadata = [_case_metadata(value, lookup) for value in fingerprints]
            metadata_frame = pd.DataFrame(metadata)
            featured.insert(1, "transaction_fingerprint", fingerprints.to_numpy())
            featured.insert(
                2,
                "source_row_number",
                raw_chunk["source_row_number"].astype("int64").to_numpy(),
            )
            for column in metadata_frame.columns:
                featured[column] = metadata_frame[column].to_numpy()
            matches = set(fingerprints[metadata_frame["pattern_matched"]])
            observed_matches.update(matches)
            times = pd.to_datetime(featured["event_time"], utc=True)
            part_min, part_max = times.min(), times.max()
            minimum = part_min if minimum is None else min(minimum, part_min)
            maximum = part_max if maximum is None else max(maximum, part_max)

            # pandas serializes attrs into Parquet schema metadata.  The feature
            # state can be very large and is runtime-only, so no attrs may cross
            # this persistence boundary.
            featured.attrs.clear()
            featured.to_parquet(
                output_partition / f"part-{output_part:06d}.parquet",
                index=False,
                compression="zstd",
            )
            output_part += 1
            files_written += 1
            written += len(featured)

    if written != max_rows:
        raise RuntimeError(f"expected to write {max_rows} rows, wrote {written}")
    return {
        "rows": written,
        "partitions": partitions,
        "parquet_files": files_written,
        "matched_fingerprints": observed_matches,
        "minimum_event_time": minimum,
        "maximum_event_time": maximum,
    }


def _case_metadata(
    fingerprint: str, lookup: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    match = lookup.get(fingerprint)
    if match is None:
        return {
            "pattern_matched": False,
            "laundering_case_ids": "[]",
            "laundering_typologies": "[]",
            "laundering_case_positions": "[]",
            "laundering_case_count": 0,
        }
    return {"pattern_matched": True, **match}


def _iso(value: pd.Timestamp | None) -> str | None:
    return None if value is None or pd.isna(value) else value.isoformat()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--patterns", type=Path, default=DEFAULT_PATTERNS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument(
        "--skip-hash-verification",
        action="store_true",
        help="allow unregistered inputs (intended only for local synthetic fixtures)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = build_feature_store(
        args.input,
        args.patterns,
        args.output,
        chunk_size=args.chunk_size,
        max_rows=args.max_rows,
        verify_hash=not args.skip_hash_verification,
        registry_path=args.registry,
    )
    print(
        f"wrote {manifest['output']['rows']:,} rows in "
        f"{manifest['output']['parquet_files']:,} Parquet files to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_feature_store", "file_sha256"]
