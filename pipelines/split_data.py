"""Create deterministic, non-overlapping chronological data splits.

All rows sharing an event timestamp stay in the same split.  Consequently the
requested 70/15/15 proportions are targets and actual counts can differ by one
timestamp group; this is preferable to leaking a simultaneous event group
across adjacent evaluation periods.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import pandas as pd
import pyarrow.parquet as pq


DEFAULT_INPUT = Path("data/processed/feature_store")
DEFAULT_OUTPUT = Path("data/processed/splits")
DEFAULT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
SPLIT_NAMES = ("train", "validation", "test")


def split_feature_store(
    input_path: str | Path = DEFAULT_INPUT,
    output_path: str | Path = DEFAULT_OUTPUT,
    *,
    ratios: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Split an ordered feature store and return its split manifest."""
    source = Path(input_path).resolve()
    destination = Path(output_path).resolve()
    proportions = _validated_ratios(ratios or DEFAULT_RATIOS)
    if not source.is_dir():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(
            f"destination already exists: {destination}; choose a new path or remove it explicitly"
        )
    files = _ordered_files(source)
    if not files:
        raise ValueError(f"no partitioned Parquet data found under {source}")
    total_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
    if total_rows < 3:
        raise ValueError("at least three rows are required for train/validation/test")

    # Clamp nominal row boundaries so even the smallest supported data set has
    # room for at least one row in every split before timestamp-group adjustment.
    train_target = min(
        max(1, int(total_rows * proportions["train"])), total_rows - 2
    )
    validation_end_target = max(
        train_target + 1,
        int(total_rows * (proportions["train"] + proportions["validation"])),
    )
    validation_end_target = min(validation_end_target, total_rows - 1)
    train_cutoff = _timestamp_at(files, train_target)
    test_cutoff = _timestamp_at(files, validation_end_target)
    if train_cutoff >= test_cutoff:
        raise ValueError(
            "dataset has too few distinct timestamps to create three non-overlapping splits"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".split-build-", dir=destination.parent
    ) as temporary:
        staged = Path(temporary) / "splits"
        staged.mkdir()
        stats = {name: _empty_stats() for name in SPLIT_NAMES}
        part_numbers = {name: 0 for name in SPLIT_NAMES}
        prior_key: tuple[pd.Timestamp, int] | None = None

        for parquet_file in files:
            frame = pq.ParquetFile(parquet_file).read().to_pandas()
            if frame.empty:
                continue
            times = pd.to_datetime(frame["event_time"], errors="raise", utc=True)
            rows = pd.to_numeric(frame["source_row_number"], errors="raise").astype("int64")
            keys = list(zip(times, rows))
            if any(left > right for left, right in zip(keys, keys[1:])):
                raise ValueError(f"feature part is not chronologically ordered: {parquet_file}")
            if prior_key is not None and keys[0] < prior_key:
                raise ValueError(f"feature parts are not globally ordered at {parquet_file}")
            prior_key = keys[-1]

            assignments = pd.Series("test", index=frame.index, dtype="object")
            assignments.loc[times < train_cutoff] = "train"
            assignments.loc[(times >= train_cutoff) & (times < test_cutoff)] = "validation"
            event_date = _event_date(parquet_file)
            for name in SPLIT_NAMES:
                subset = frame.loc[assignments == name]
                if subset.empty:
                    continue
                partition = staged / name / f"event_date={event_date}"
                partition.mkdir(parents=True, exist_ok=True)
                output_file = partition / f"part-{part_numbers[name]:06d}.parquet"
                subset.attrs.clear()
                subset.to_parquet(output_file, index=False, compression="zstd")
                part_numbers[name] += 1
                _update_stats(stats[name], subset)

        if any(stats[name]["rows"] == 0 for name in SPLIT_NAMES):
            raise ValueError("timestamp-aware boundaries produced an empty split")
        _validate_no_overlap(stats, total_rows)
        manifest = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_feature_store": str(source),
            "strategy": "chronological_timestamp_groups",
            "target_ratios": proportions,
            "total_rows": total_rows,
            "cutoffs": {
                "validation_starts_at": train_cutoff.isoformat(),
                "test_starts_at": test_cutoff.isoformat(),
            },
            "splits": {
                name: {
                    "rows": stats[name]["rows"],
                    "actual_ratio": stats[name]["rows"] / total_rows,
                    "minimum_event_time": stats[name]["minimum_event_time"].isoformat(),
                    "maximum_event_time": stats[name]["maximum_event_time"].isoformat(),
                    "parquet_files": part_numbers[name],
                }
                for name in SPLIT_NAMES
            },
        }
        (staged / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staged.replace(destination)
    return manifest


def _validated_ratios(ratios: Mapping[str, float]) -> dict[str, float]:
    if set(ratios) != set(SPLIT_NAMES):
        raise ValueError(f"ratios must contain exactly {list(SPLIT_NAMES)}")
    converted = {name: float(ratios[name]) for name in SPLIT_NAMES}
    if any(value <= 0 or value >= 1 for value in converted.values()):
        raise ValueError("each split ratio must be between zero and one")
    if abs(sum(converted.values()) - 1.0) > 1e-9:
        raise ValueError("split ratios must sum to one")
    return converted


def _ordered_files(source: Path) -> list[Path]:
    return sorted(
        path
        for path in source.glob("event_date=*/part-*.parquet")
        if path.is_file()
    )


def _timestamp_at(files: Sequence[Path], zero_based_index: int) -> pd.Timestamp:
    remaining = zero_based_index
    for path in files:
        count = pq.ParquetFile(path).metadata.num_rows
        if remaining < count:
            column = pq.ParquetFile(path).read(columns=["event_time"]).to_pandas()
            return pd.to_datetime(column["event_time"].iloc[remaining], utc=True)
        remaining -= count
    raise IndexError(zero_based_index)


def _event_date(path: Path) -> str:
    parent = path.parent.name
    if not parent.startswith("event_date=") or len(parent) <= len("event_date="):
        raise ValueError(f"expected Hive event_date partition for {path}")
    return parent.split("=", 1)[1]


def _empty_stats() -> dict[str, Any]:
    return {"rows": 0, "minimum_event_time": None, "maximum_event_time": None}


def _update_stats(stats: dict[str, Any], frame: pd.DataFrame) -> None:
    times = pd.to_datetime(frame["event_time"], utc=True)
    minimum, maximum = times.min(), times.max()
    stats["rows"] += len(frame)
    stats["minimum_event_time"] = (
        minimum
        if stats["minimum_event_time"] is None
        else min(stats["minimum_event_time"], minimum)
    )
    stats["maximum_event_time"] = (
        maximum
        if stats["maximum_event_time"] is None
        else max(stats["maximum_event_time"], maximum)
    )


def _validate_no_overlap(stats: Mapping[str, Mapping[str, Any]], total: int) -> None:
    if sum(int(stats[name]["rows"]) for name in SPLIT_NAMES) != total:
        raise RuntimeError("split row counts do not preserve the source row count")
    if not (
        stats["train"]["maximum_event_time"] < stats["validation"]["minimum_event_time"]
        and stats["validation"]["maximum_event_time"] < stats["test"]["minimum_event_time"]
    ):
        raise RuntimeError("chronological split timestamp ranges overlap")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train", type=float, default=DEFAULT_RATIOS["train"])
    parser.add_argument(
        "--validation", type=float, default=DEFAULT_RATIOS["validation"]
    )
    parser.add_argument("--test", type=float, default=DEFAULT_RATIOS["test"])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = split_feature_store(
        args.input,
        args.output,
        ratios={"train": args.train, "validation": args.validation, "test": args.test},
    )
    counts = ", ".join(
        f"{name}={manifest['splits'][name]['rows']:,}" for name in SPLIT_NAMES
    )
    print(f"created chronological splits: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["split_feature_store"]
