from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from pipelines.build_feature_store import build_feature_store
from pipelines.split_data import split_feature_store


HEADER = (
    "Timestamp,From Bank,Account,To Bank,Account,Amount Received,"
    "Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering"
)


def _row(timestamp: str, amount: str, label: str = "0", destination: str = "D") -> str:
    return (
        f"{timestamp},001,S,{destination},R,{amount},US Dollar,{amount},"
        f"US Dollar,ACH,{label}"
    )


def test_builder_orders_before_limiting_and_keeps_equal_times_causal(tmp_path: Path) -> None:
    rows = [
        _row("2022/09/01 02:00", "40"),
        _row("2022/09/01 00:00", "10"),
        _row("2022/09/02 00:00", "60"),
        _row("2022/09/01 00:00", "20", label="1", destination="X"),
        _row("2022/09/01 01:00", "30"),
        _row("2022/09/01 03:00", "50"),
    ]
    source = tmp_path / "HI-Small_Trans.csv"
    source.write_text(HEADER + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    pattern = tmp_path / "HI-Small_Patterns.txt"
    pattern.write_text(
        "BEGIN LAUNDERING ATTEMPT - RANDOM: Max 1 hops\n"
        + rows[3]
        + "\nEND LAUNDERING ATTEMPT - RANDOM\n",
        encoding="utf-8",
    )
    output = tmp_path / "features"

    manifest = build_feature_store(
        source,
        pattern,
        output,
        chunk_size=1,
        max_rows=4,
        verify_hash=False,
    )

    files = sorted(output.glob("event_date=*/part-*.parquet"))
    result = pd.concat(
        [pq.ParquetFile(path).read().to_pandas() for path in files], ignore_index=True
    )
    times = pd.to_datetime(result["event_time"], utc=True)
    assert times.is_monotonic_increasing
    assert result["amount"].tolist() == [10.0, 20.0, 30.0, 40.0]
    assert result.loc[:1, "source_txn_count_1h"].tolist() == [0, 0]
    assert result.loc[2, "source_txn_count_1h"] == 2
    assert result["event_id"].is_unique
    assert result["pattern_matched"].tolist() == [False, True, False, False]
    assert "RANDOM" in result.loc[1, "laundering_typologies"]
    assert manifest["source"]["physical_rows_scanned"] == 6
    assert manifest["output"]["rows"] == 4
    assert manifest["ordering"]["max_rows_semantics"].startswith("earliest")
    metadata = pq.read_metadata(files[0]).metadata or {}
    assert b"PANDAS_ATTRS" not in metadata


def test_builder_refuses_to_replace_existing_destination(tmp_path: Path) -> None:
    source = tmp_path / "HI-Small_Trans.csv"
    source.write_text(HEADER + "\n" + _row("2022/09/01 00:00", "1") + "\n")
    pattern = tmp_path / "HI-Small_Patterns.txt"
    pattern.write_text(
        "BEGIN LAUNDERING ATTEMPT - RANDOM\n"
        + _row("2022/09/01 00:00", "1", label="1")
        + "\nEND LAUNDERING ATTEMPT - RANDOM\n"
    )
    output = tmp_path / "already-there"
    output.mkdir()
    with pytest.raises(FileExistsError):
        build_feature_store(source, pattern, output, verify_hash=False)


def _write_feature_part(path: Path, times: list[str], first_row: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "event_id": [f"event-{number}" for number in range(first_row, first_row + len(times))],
            "source_row_number": range(first_row, first_row + len(times)),
            "event_time": pd.to_datetime(times, utc=True),
            "amount": range(len(times)),
        }
    )
    frame.to_parquet(path, index=False)


def test_split_is_deterministic_chronological_and_keeps_timestamp_groups(tmp_path: Path) -> None:
    source = tmp_path / "feature-store"
    day_one = [f"2022-09-01 {hour:02d}:00" for hour in range(6)]
    # Two events at hour 08 straddle the nominal 70% row boundary.
    day_two = [
        "2022-09-02 06:00",
        "2022-09-02 07:00",
        "2022-09-02 08:00",
        "2022-09-02 08:00",
        "2022-09-02 09:00",
        "2022-09-02 10:00",
        "2022-09-02 11:00",
    ]
    _write_feature_part(source / "event_date=2022-09-01" / "part-000000.parquet", day_one, 1)
    _write_feature_part(source / "event_date=2022-09-02" / "part-000001.parquet", day_two, 7)

    manifest = split_feature_store(source, tmp_path / "splits")

    split_frames = {}
    for name in ("train", "validation", "test"):
        files = sorted((tmp_path / "splits" / name).glob("event_date=*/part-*.parquet"))
        split_frames[name] = pd.concat(
            [pq.ParquetFile(path).read().to_pandas() for path in files], ignore_index=True
        )
    ids = [set(frame["event_id"]) for frame in split_frames.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert sum(map(len, ids)) == 13
    train_max = pd.to_datetime(split_frames["train"]["event_time"], utc=True).max()
    validation = pd.to_datetime(split_frames["validation"]["event_time"], utc=True)
    test_min = pd.to_datetime(split_frames["test"]["event_time"], utc=True).min()
    assert train_max < validation.min()
    assert validation.max() < test_min
    assert not (
        set(pd.to_datetime(split_frames["train"]["event_time"], utc=True))
        & set(validation)
    )
    assert sum(item["rows"] for item in manifest["splits"].values()) == 13


def test_split_rejects_invalid_ratios(tmp_path: Path) -> None:
    source = tmp_path / "feature-store"
    _write_feature_part(
        source / "event_date=2022-09-01" / "part-000000.parquet",
        ["2022-09-01 00:00", "2022-09-01 01:00", "2022-09-01 02:00"],
        1,
    )
    with pytest.raises(ValueError, match="sum to one"):
        split_feature_store(
            source,
            tmp_path / "splits",
            ratios={"train": 0.7, "validation": 0.2, "test": 0.2},
        )


def test_split_supports_three_distinct_rows(tmp_path: Path) -> None:
    source = tmp_path / "feature-store"
    _write_feature_part(
        source / "event_date=2022-09-01" / "part-000000.parquet",
        ["2022-09-01 00:00", "2022-09-01 01:00", "2022-09-01 02:00"],
        1,
    )
    manifest = split_feature_store(source, tmp_path / "splits")
    assert [manifest["splits"][name]["rows"] for name in ("train", "validation", "test")] == [1, 1, 1]
