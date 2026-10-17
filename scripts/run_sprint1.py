#!/usr/bin/env python3
"""Run the complete Sprint 1 data path with non-destructive defaults.

An unbounded run uses the publication paths documented in the repository.  A
bounded ``--max-rows`` run is automatically isolated under a named development
run so it cannot be confused with, or overwrite, publication artifacts.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any, Mapping, Sequence

import numpy
import pandas
import pyarrow


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.case_queue import QueueConfig  # noqa: E402
from pipelines.build_case_queue import read_parquet_frame, write_queue_artifacts  # noqa: E402
from pipelines.build_feature_store import build_feature_store  # noqa: E402
from pipelines.preprocess_splits import preprocess_splits, resolve_split_paths  # noqa: E402
from pipelines.split_data import split_feature_store  # noqa: E402
from scripts.acquire_data import load_registry, verify_dataset  # noqa: E402
from scripts.build_data_dictionary import (  # noqa: E402
    build_rows,
    profile as profile_dictionary,
    write_dictionary,
)
from scripts.run_eda import run_eda  # noqa: E402
from scripts.verify_sprint1 import (  # noqa: E402
    ArtifactLayout,
    REQUIRED_RAW_FILES,
    layout_for,
    sha256_file,
    verify_sprint1,
    write_verification_report,
)


DATASET_ID = "ibm_aml_hi_small"


def output_targets(layout: ArtifactLayout) -> tuple[Path, ...]:
    """Return every path the orchestrator may create."""
    return (
        layout.feature_store,
        layout.splits,
        layout.model_matrix,
        layout.case_queue,
        layout.eda,
        layout.data_dictionary,
        layout.run_manifest,
        layout.verification_report,
    )


def preflight_outputs(layout: ArtifactLayout) -> None:
    """Fail before work starts if any run output already exists."""
    existing = [path for path in output_targets(layout) if path.exists()]
    if existing:
        rendered = "\n  - ".join(str(path) for path in existing)
        raise FileExistsError(
            "refusing to overwrite existing Sprint 1 artifacts:\n  - " + rendered
            + "\nChoose a new --run-name or remove the intended artifacts explicitly."
        )


def run_sprint1(
    layout: ArtifactLayout,
    *,
    max_rows: int | None = None,
    chunk_size: int | None = None,
    batch_size: int = 200_000,
) -> dict[str, Any]:
    """Execute all Sprint 1 stages and return the newly written run manifest."""
    if max_rows is not None and max_rows < 3:
        raise ValueError("max_rows must be at least three so every split can be non-empty")
    if chunk_size is not None and chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    preflight_outputs(layout)

    root = layout.project_root
    pipeline_config = _load_object(root / "config" / "pipeline.json")
    study_config = _load_object(root / "config" / "study.json")
    registry_path = root / "config" / "datasets.json"
    registry = load_registry(registry_path)
    raw_root = root / "data" / "raw"
    if not verify_dataset(
        registry,
        DATASET_ID,
        raw_root,
        selected_files=REQUIRED_RAW_FILES,
    ):
        raise RuntimeError("governed raw-data verification failed; no outputs were created")

    dataset_root = raw_root / DATASET_ID
    transactions = dataset_root / "HI-Small_Trans.csv"
    patterns = dataset_root / "HI-Small_Patterns.txt"
    effective_chunk_size = int(chunk_size or pipeline_config.get("chunk_size", 100_000))
    ratios = pipeline_config.get("split_ratios", {})

    started = datetime.now(timezone.utc)
    feature_manifest = build_feature_store(
        transactions,
        patterns,
        layout.feature_store,
        chunk_size=effective_chunk_size,
        max_rows=max_rows,
        verify_hash=True,
        registry_path=registry_path,
    )
    split_manifest = split_feature_store(
        layout.feature_store,
        layout.splits,
        ratios={
            "train": float(ratios.get("train", 0.70)),
            "validation": float(ratios.get("validation", 0.15)),
            "test": float(ratios.get("test", 0.15)),
        },
    )
    split_paths = resolve_split_paths(layout.splits)
    model_manifest = preprocess_splits(
        split_paths["train"],
        split_paths["validation"],
        split_paths["test"],
        layout.model_matrix,
    )

    eda_summary = run_eda(
        layout.feature_store,
        layout.eda,
        batch_size,
        max_rows,
        patterns,
    )
    dictionary_rows_profiled, observed = profile_dictionary(
        layout.feature_store, batch_size, max_rows
    )
    dictionary_rows = build_rows(observed, layout.feature_store, True)
    write_dictionary(dictionary_rows, layout.data_dictionary)

    test_frame, test_files = read_parquet_frame(split_paths["test"])
    queue_policy = study_config.get("queue_policy", {})
    pipeline_queue = pipeline_config.get("queue", {})
    queue_config = QueueConfig(
        capacity_per_hour=int(queue_policy.get("capacity_per_hour", pipeline_queue.get("capacity_per_hour", 25))),
        study_seed=int(pipeline_queue.get("study_seed", 20261007)),
        study_arms=tuple(study_config.get("arms", pipeline_queue.get("conditions", ()))),
    )
    queue_manifest = write_queue_artifacts(
        test_frame,
        layout.case_queue,
        config=queue_config,
        input_files=test_files,
        force=False,
    )

    verification = verify_sprint1(layout)
    write_verification_report(verification, layout.verification_report)
    finished = datetime.now(timezone.utc)
    persisted_manifests = {
        "feature_store": layout.feature_store / "manifest.json",
        "splits": layout.splits / "manifest.json",
        "model_matrix": layout.model_matrix / "manifest.json",
        "case_queue": layout.case_queue / "manifest.json",
        "eda": layout.eda / "summary.json",
        "verification": layout.verification_report,
    }
    run_manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "sprint": 1,
        "started_at_utc": started.isoformat(),
        "finished_at_utc": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
        "run_name": _run_name(layout),
        "development_sample": max_rows is not None,
        "parameters": {
            "max_rows": max_rows,
            "chunk_size": effective_chunk_size,
            "batch_size": batch_size,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "source": {
            "dataset_id": DATASET_ID,
            "transactions": str(transactions.resolve()),
            "patterns": str(patterns.resolve()),
            "raw_manifest": str((dataset_root / "manifest.json").resolve()),
        },
        "outputs": {
            "feature_store": str(layout.feature_store),
            "splits": str(layout.splits),
            "model_matrix": str(layout.model_matrix),
            "case_queue": str(layout.case_queue),
            "eda": str(layout.eda),
            "data_dictionary": str(layout.data_dictionary),
            "verification_report": str(layout.verification_report),
        },
        "stage_results": {
            "feature_rows": int(feature_manifest["output"]["rows"]),
            "split_rows": {
                name: int(split_manifest["splits"][name]["rows"])
                for name in ("train", "validation", "test")
            },
            "preprocessor_fit_rows": int(model_manifest["fit_rows"]),
            "eda_rows": int(eda_summary["profile_scope"]["rows"]),
            "dictionary_profile_rows": int(dictionary_rows_profiled),
            "dictionary_entries": len(dictionary_rows),
            "reviewer_alert_rows": int(queue_manifest["reviewer_alert_rows"]),
            "verification_status": verification["status"],
        },
        "manifests": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for name, path in persisted_manifests.items()
        },
    }
    _write_new_json(layout.run_manifest, run_manifest)
    return run_manifest


def _run_name(layout: ArtifactLayout) -> str | None:
    parts = layout.feature_store.parts
    try:
        index = parts.index("runs")
    except ValueError:
        return None
    return parts[index + 1] if index + 1 < len(parts) else None


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read configuration {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a JSON object: {path}")
    return value


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite run manifest: {path}") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--run-name",
        help=(
            "Isolate outputs below data/processed/runs and reports/runs. "
            "With --max-rows, defaults to dev-<rows>."
        ),
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        help="Build only the globally earliest N events as an explicitly marked development run",
    )
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--batch-size", type=int, default=200_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_rows is not None and args.max_rows < 3:
        raise SystemExit("--max-rows must be at least 3")
    run_name = args.run_name
    if args.max_rows is not None and run_name is None:
        run_name = f"dev-{args.max_rows}"
    layout = layout_for(args.project_root, run_name)
    manifest = run_sprint1(
        layout,
        max_rows=args.max_rows,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
    )
    print(
        f"Sprint 1 complete and verified: {manifest['stage_results']['feature_rows']:,} rows; "
        f"run manifest {layout.run_manifest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["output_targets", "preflight_outputs", "run_sprint1"]
