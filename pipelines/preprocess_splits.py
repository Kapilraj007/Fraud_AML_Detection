"""Fit preprocessing on the chronological train split and freeze all others."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backend.app.ml.preprocessing import FeaturePreprocessor


DEFAULT_METADATA_COLUMNS = (
    "event_id", "transaction_id", "transaction_fingerprint", "source_row_number",
    "dataset", "time_semantics", "event_time", "relative_time_seconds",
    "device_info", "device_type", "label_name", "label", "case_id", "typology", "case_detail",
    "pattern_detail", "case_position", "pattern_line_number", "pattern_id",
    "pattern_position", "pattern_matched", "laundering_case_ids",
    "laundering_typologies", "laundering_case_positions",
    "laundering_case_count",
)


def preprocess_splits(
    train_path: str | Path,
    validation_path: str | Path,
    test_path: str | Path,
    output_root: str | Path,
    *,
    state_path: str | Path | None = None,
) -> dict[str, Any]:
    """Create split model matrices using state fitted exclusively on train."""
    inputs = {
        "train": Path(train_path),
        "validation": Path(validation_path),
        "test": Path(test_path),
    }
    frames = {name: _read_parquet(path) for name, path in inputs.items()}
    preprocessor = FeaturePreprocessor().fit(frames["train"])

    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    fitted_state_path = (Path(state_path) if state_path is not None
                         else output / "preprocessor.json")
    preprocessor.save(fitted_state_path)
    state_bytes = fitted_state_path.read_bytes()

    split_records: dict[str, dict[str, Any]] = {}
    frozen_state = preprocessor.to_dict()
    for name in ("train", "validation", "test"):
        original = frames[name]
        features = preprocessor.transform(original).reset_index(drop=True)
        if preprocessor.to_dict() != frozen_state:
            raise RuntimeError("transform unexpectedly mutated fitted state")
        metadata_columns = [column for column in DEFAULT_METADATA_COLUMNS
                            if column in original.columns]
        metadata = original.loc[:, metadata_columns].reset_index(drop=True)
        matrix = pd.concat([metadata, features], axis=1)
        destination = output / f"{name}.parquet"
        matrix.to_parquet(destination, index=False, compression="zstd")
        split_records[name] = {
            "input": str(inputs[name].resolve()),
            "output": str(destination.resolve()),
            "rows": len(matrix),
            "metadata_columns": metadata_columns,
        }

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "fit_split": "train",
        "fit_rows": len(frames["train"]),
        "state_path": str(fitted_state_path.resolve()),
        "state_sha256": hashlib.sha256(state_bytes).hexdigest(),
        "feature_columns": list(preprocessor.feature_columns),
        "splits": split_records,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def resolve_split_paths(root: str | Path) -> dict[str, Path]:
    """Resolve common file, directory, and Hive split layouts."""
    base = Path(root)
    return {
        "train": _resolve_one(base, ("train",)),
        "validation": _resolve_one(base, ("validation", "val")),
        "test": _resolve_one(base, ("test",)),
    }


def _resolve_one(root: Path, names: tuple[str, ...]) -> Path:
    candidates: list[Path] = []
    for name in names:
        candidates.extend((root / f"{name}.parquet", root / name,
                           root / f"split={name}"))
    for candidate in candidates:
        if candidate.is_file() or (
            candidate.is_dir() and any(candidate.rglob("*.parquet"))
        ):
            return candidate
    rendered = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"could not resolve split; checked: {rendered}")


def _read_parquet(path: Path) -> pd.DataFrame:
    if path.is_file():
        return pd.read_parquet(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files below {path}")
    frames = [pd.read_parquet(file) for file in files]
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root", type=Path, default=Path("data/processed/splits")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/model_matrix")
    )
    parser.add_argument("--state-output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = resolve_split_paths(args.input_root)
    manifest = preprocess_splits(
        paths["train"], paths["validation"], paths["test"], args.output_root,
        state_path=args.state_output,
    )
    print(json.dumps({
        "fit_split": manifest["fit_split"],
        "fit_rows": manifest["fit_rows"],
        "feature_count": len(manifest["feature_columns"]),
        "output_root": str(args.output_root.resolve()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["preprocess_splits", "resolve_split_paths"]
