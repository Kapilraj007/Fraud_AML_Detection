"""Materialise a capacity-constrained analyst queue and sealed evaluation key.

The reviewer-facing Parquet file never contains the ground-truth label.  The
evaluation key is written separately so that offline queue metrics can be
computed without making labels available to an analyst interface.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from backend.app.services.case_queue import (
    QueueConfig,
    build_case_queue,
    evaluate_queue,
)


DEFAULT_INPUT = Path("data/processed/splits/test")
DEFAULT_OUTPUT = Path("data/processed/case_queue")


def _parquet_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.casefold() != ".parquet":
            raise ValueError(f"input file must be Parquet: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"input does not exist: {path}")
    files = sorted(candidate for candidate in path.rglob("*.parquet") if candidate.is_file())
    if not files:
        raise FileNotFoundError(f"no Parquet files found under: {path}")
    return files


def read_parquet_frame(path: str | Path) -> tuple[pd.DataFrame, list[Path]]:
    """Read one Parquet file or a partition tree deterministically."""
    root = Path(path)
    files = _parquet_files(root)
    parts = [pd.read_parquet(file) for file in files]
    frame = parts[0] if len(parts) == 1 else pd.concat(parts, ignore_index=True)
    return frame, files


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return _json_value(value.item())
    return value


def write_queue_artifacts(
    frame: pd.DataFrame,
    output_dir: str | Path,
    *,
    config: QueueConfig | None = None,
    input_files: Sequence[Path] = (),
    force: bool = False,
) -> dict[str, Any]:
    """Build queue outputs and return the JSON-serialisable manifest."""
    output = Path(output_dir)
    alerts_path = output / "analyst_alerts.parquet"
    key_path = output / "sealed_evaluation_key.parquet"
    metrics_path = output / "queue_metrics.json"
    manifest_path = output / "manifest.json"
    targets = (alerts_path, key_path, metrics_path, manifest_path)
    existing = [path for path in targets if path.exists()]
    if existing and not force:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite queue artifacts: {names}")

    cfg = config or QueueConfig()
    bundle = build_case_queue(frame, cfg)
    metrics = {
        key: _json_value(value)
        for key, value in evaluate_queue(
            bundle, minutes_per_review=cfg.minutes_per_review
        ).items()
    }
    output.mkdir(parents=True, exist_ok=True)
    bundle.alerts.to_parquet(alerts_path, index=False)
    bundle.evaluation_key.to_parquet(key_path, index=False)
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    source_files = []
    for path in input_files:
        source_files.append({
            "path": path.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "queue_config": asdict(cfg),
        "input_rows": int(len(frame)),
        "reviewer_alert_rows": int(len(bundle.alerts)),
        "evaluation_key_rows": int(len(bundle.evaluation_key)),
        "source_files": source_files,
        "outputs": {
            "analyst_alerts": alerts_path.name,
            "sealed_evaluation_key": key_path.name,
            "queue_metrics": metrics_path.name,
        },
        "label_boundary": (
            "analyst_alerts.parquet excludes ground truth; "
            "sealed_evaluation_key.parquet is evaluation-only"
        ),
        "metrics": metrics,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--capacity-per-hour", type=int, default=25)
    parser.add_argument("--score-column")
    parser.add_argument("--study-seed", type=int, default=20261007)
    parser.add_argument("--minutes-per-review", type=float, default=5.0)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    frame, files = read_parquet_frame(args.input)
    config = QueueConfig(
        capacity_per_hour=args.capacity_per_hour,
        score_column=args.score_column,
        study_seed=args.study_seed,
        minutes_per_review=args.minutes_per_review,
    )
    manifest = write_queue_artifacts(
        frame, args.output, config=config, input_files=files, force=args.force
    )
    print(json.dumps(manifest["metrics"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

