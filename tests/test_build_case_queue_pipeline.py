from __future__ import annotations

import json

import pandas as pd
import pytest

from backend.app.services.case_queue import QueueConfig
from pipelines.build_case_queue import read_parquet_frame, write_queue_artifacts


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "event_id": ["a", "b", "c", "d"],
        "event_time": pd.to_datetime([
            "2026-01-01 00:01Z", "2026-01-01 00:02Z",
            "2026-01-01 01:01Z", "2026-01-01 01:02Z",
        ]),
        "amount": [10.0, 20.0, 30.0, 40.0],
        "risk": [0.1, 0.9, 0.8, 0.2],
        "label": [0, 1, 1, 0],
        "case_id": [None, "case-1", "case-2", None],
    })


def test_materialised_alerts_are_label_safe_and_reproducible(tmp_path) -> None:
    source = tmp_path / "source.parquet"
    _frame().to_parquet(source, index=False)
    loaded, files = read_parquet_frame(source)
    output = tmp_path / "queue"
    manifest = write_queue_artifacts(
        loaded,
        output,
        config=QueueConfig(capacity_per_hour=1, score_column="risk", study_seed=4),
        input_files=files,
    )

    alerts = pd.read_parquet(output / "analyst_alerts.parquet")
    key = pd.read_parquet(output / "sealed_evaluation_key.parquet")
    assert alerts["event_id"].tolist() == ["b", "c"]
    assert "label" not in alerts.columns
    assert "case_id" not in alerts.columns
    assert key["label"].tolist() == [1, 0, 1, 0]
    assert manifest["reviewer_alert_rows"] == 2
    assert manifest["source_files"][0]["sha256"]
    assert json.loads((output / "queue_metrics.json").read_text())["alert_recall"] == 1.0


def test_queue_artifacts_require_force_to_overwrite(tmp_path) -> None:
    output = tmp_path / "queue"
    config = QueueConfig(capacity_per_hour=1, score_column="risk")
    write_queue_artifacts(_frame(), output, config=config)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_queue_artifacts(_frame(), output, config=config)
    write_queue_artifacts(_frame(), output, config=config, force=True)


def test_read_parquet_tree_has_stable_file_order(tmp_path) -> None:
    root = tmp_path / "parts"
    root.mkdir()
    _frame().iloc[2:].to_parquet(root / "b.parquet", index=False)
    _frame().iloc[:2].to_parquet(root / "a.parquet", index=False)
    frame, files = read_parquet_frame(root)
    assert [path.name for path in files] == ["a.parquet", "b.parquet"]
    assert frame["event_id"].tolist() == ["a", "b", "c", "d"]
