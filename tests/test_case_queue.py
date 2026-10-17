from __future__ import annotations

import pandas as pd
import pytest

from backend.app.services.case_queue import (
    QueueConfig,
    build_case_queue,
    evaluate_queue,
)


def _frame() -> pd.DataFrame:
    frame = pd.DataFrame({
        "event_id": [f"e{i}" for i in range(8)],
        "event_time": pd.to_datetime([
            "2026-01-01 00:01Z", "2026-01-01 00:02Z",
            "2026-01-01 00:03Z", "2026-01-01 00:04Z",
            "2026-01-01 01:01Z", "2026-01-01 01:02Z",
            "2026-01-01 01:03Z", "2026-01-01 01:04Z",
        ]),
        "relative_time_seconds": list(range(8)),
        "amount": [10, 20, 30, 40, 50, 60, 70, 80],
        "risk": [0.5, 0.9, 0.9, 0.1, 0.8, 0.7, 0.6, 0.5],
        "label": [0, 1, 0, 0, 1, 0, 1, 0],
        "case_id": [None, "c1", None, None, "c2", None, "c2", None],
        "typology": [None, "fan-out", None, None, "cycle", None, "cycle", None],
    })
    frame.attrs["label_column"] = "label"
    return frame


def test_capacity_ties_and_public_label_separation_are_deterministic() -> None:
    config = QueueConfig(capacity_per_hour=2, score_column="risk", study_seed=7)
    one = build_case_queue(_frame(), config)
    two = build_case_queue(_frame().sample(frac=1, random_state=5), config)
    assert one.alerts.groupby("alert_hour").size().tolist() == [2, 2]
    assert one.alerts["event_id"].tolist() == ["e1", "e2", "e4", "e5"]
    assert one.alerts["event_id"].tolist() == two.alerts["event_id"].tolist()
    assert "label" not in one.alerts
    assert "label" in one.evaluation_key


@pytest.mark.parametrize("column", ["label", "isFraud", "Is Laundering", "target"])
def test_ground_truth_score_is_rejected(column: str) -> None:
    frame = _frame().rename(columns={"label": column})
    with pytest.raises(ValueError, match="ground truth"):
        build_case_queue(frame, QueueConfig(score_column=column))


def test_label_changes_never_change_ranking_or_development_score() -> None:
    first = _frame().drop(columns="risk")
    second = first.copy()
    second["label"] = 1 - second["label"]
    left = build_case_queue(first, QueueConfig(capacity_per_hour=3))
    right = build_case_queue(second, QueueConfig(capacity_per_hour=3))
    pd.testing.assert_frame_equal(left.alerts, right.alerts)


def test_arm_assignment_is_balanced_and_case_consistent() -> None:
    bundle = build_case_queue(_frame(), QueueConfig(capacity_per_hour=4, score_column="risk"))
    key = bundle.evaluation_key.copy()
    key["unit"] = key["case_id"].where(key["case_id"].notna(), key["event_id"])
    counts = key.drop_duplicates("unit")["study_arm"].value_counts()
    assert counts.max() - counts.min() <= 1
    c2 = key[key["case_id"].eq("c2")]
    assert c2["study_arm"].nunique() == 1


def test_queue_metrics_include_alert_case_capacity_and_value() -> None:
    bundle = build_case_queue(
        _frame(), QueueConfig(capacity_per_hour=2, score_column="risk")
    )
    metrics = evaluate_queue(bundle, minutes_per_review=5)
    assert metrics["alerts_reviewed"] == 4
    assert metrics["alert_precision"] == pytest.approx(0.5)
    assert metrics["alert_recall"] == pytest.approx(2 / 3)
    assert metrics["case_recall"] == pytest.approx(1.0)
    assert metrics["false_escalations"] == 2
    assert metrics["reviewed_per_hour"] == pytest.approx(2.0)
    assert metrics["laundering_value_per_analyst_hour"] == pytest.approx(210.0)


def test_pattern_ground_truth_is_sealed_but_drives_case_evaluation() -> None:
    frame = _frame().drop(columns=["case_id", "typology"])
    frame["laundering_case_ids"] = [
        "[]", '["c1"]', "[]", "[]", '["c2"]', "[]", '["c2","c3"]', "[]"
    ]
    frame["laundering_typologies"] = [
        "[]", '["FAN-OUT"]', "[]", "[]", '["CYCLE"]', "[]",
        '["CYCLE","FAN-IN"]', "[]",
    ]
    frame["pattern_matched"] = frame["label"].astype(bool)
    bundle = build_case_queue(
        frame, QueueConfig(capacity_per_hour=2, score_column="risk")
    )

    assert not {
        "label", "laundering_case_ids", "laundering_typologies", "pattern_matched"
    } & set(bundle.alerts)
    assert "case_ids" in bundle.evaluation_key
    assert "typologies" in bundle.evaluation_key
    metrics = evaluate_queue(bundle)
    assert metrics["case_recall"] == pytest.approx(2 / 3)
