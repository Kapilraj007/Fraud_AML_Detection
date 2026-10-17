"""Versioned, checksummed model bundles for trusted Sprint 2 artifacts.

The LightGBM primary model is stored in its native text format.  JSON is used
for the feature schema, calibration, metrics, and portable logistic baseline.
Isolation Forest is an explicitly secondary trusted-local joblib artifact;
loading it requires an opt-in flag after checksum verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from backend.app.ml.model import (
    IsolationForestSignal,
    LightGBMModel,
    LinearLogitModel,
    ModelDependencyError,
    PlattCalibrator,
)


BUNDLE_SCHEMA_VERSION = 1
PRIMARY_MODEL_FILE = "model.txt"
MANIFEST_FILE = "manifest.json"


class ArtifactError(RuntimeError):
    """Raised when a model bundle is incomplete, corrupt, or incompatible."""


@dataclass(frozen=True)
class VerifiedBundle:
    root: Path
    manifest: dict[str, Any]

    def path(self, logical_name: str) -> Path:
        try:
            relative = self.manifest["files"][logical_name]["path"]
        except (KeyError, TypeError) as error:
            raise ArtifactError(f"bundle has no file named {logical_name!r}") from error
        return _safe_child(self.root, relative)


@dataclass
class LoadedModelBundle:
    model: LightGBMModel
    manifest: dict[str, Any]
    feature_schema: dict[str, Any]
    metrics: dict[str, Any]
    training: dict[str, Any]
    logistic_baseline: LinearLogitModel | None = None
    anomaly_signal: IsolationForestSignal | None = None


def sha256_file(path: str | Path, *, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def save_model_bundle(
    output_directory: str | Path,
    model: LightGBMModel,
    *,
    feature_dtypes: Mapping[str, str] | None = None,
    metrics: Mapping[str, Any] | None = None,
    training: Mapping[str, Any] | None = None,
    configuration: Mapping[str, Any] | None = None,
    calibrator: PlattCalibrator | None = None,
    logistic_baseline: LinearLogitModel | None = None,
    anomaly_signal: IsolationForestSignal | None = None,
    xgboost_crosscheck: Any | None = None,
    preprocessor_path: str | Path | None = None,
    bundle_id: str | None = None,
) -> Path:
    """Create a model bundle atomically and refuse to overwrite an existing path."""
    destination = Path(output_directory).resolve()
    if destination.exists():
        raise FileExistsError(f"model bundle destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).isoformat()
    identity = bundle_id or datetime.now(timezone.utc).strftime("aml-%Y%m%dT%H%M%SZ")

    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}-build-", dir=destination.parent
    ) as temporary:
        staged = Path(temporary) / destination.name
        staged.mkdir()
        model_path = staged / PRIMARY_MODEL_FILE
        model.booster.save_model(str(model_path), num_iteration=model.best_iteration)
        if not model_path.is_file() or model_path.stat().st_size == 0:
            raise ArtifactError("LightGBM did not emit a non-empty native model")

        calibration = calibrator or model.calibrator or PlattCalibrator()
        schema = {
            "schema_version": 1,
            "feature_names": list(model.feature_names),
            "feature_dtypes": {
                name: str((feature_dtypes or {}).get(name, "float64"))
                for name in model.feature_names
            },
            "feature_count": len(model.feature_names),
            "reject_non_finite": True,
        }
        _write_json(staged / "calibration.json", calibration.to_dict())
        _write_json(staged / "feature_schema.json", schema)
        _write_json(staged / "metrics.json", dict(metrics or {}))
        _write_json(staged / "training.json", dict(training or {}))
        _write_json(staged / "configuration.json", dict(configuration or {}))

        file_records: dict[str, dict[str, Any]] = {}
        _record(file_records, "primary_model", staged, model_path, "lightgbm-native-text")
        for logical, filename in (
            ("calibration", "calibration.json"),
            ("feature_schema", "feature_schema.json"),
            ("metrics", "metrics.json"),
            ("training", "training.json"),
            ("configuration", "configuration.json"),
        ):
            _record(file_records, logical, staged, staged / filename, "json")

        if logistic_baseline is not None:
            baseline_path = staged / "logistic_baseline.json"
            _write_json(baseline_path, logistic_baseline.to_dict())
            _record(file_records, "logistic_baseline", staged, baseline_path, "json")

        if anomaly_signal is not None:
            try:
                import joblib
            except ImportError as error:  # pragma: no cover - dependency path
                raise ModelDependencyError(
                    "joblib/scikit-learn is required to save Isolation Forest"
                ) from error
            anomaly_path = staged / "isolation_forest.joblib"
            joblib.dump(
                {
                    "kind": "isolation-forest-signal",
                    "version": 1,
                    "estimator": anomaly_signal.estimator,
                    "feature_names": list(anomaly_signal.feature_names),
                    "center": anomaly_signal.center,
                    "scale": anomaly_signal.scale,
                },
                anomaly_path,
                compress=3,
            )
            _record(
                file_records,
                "anomaly_signal",
                staged,
                anomaly_path,
                "trusted-local-joblib",
            )

        if xgboost_crosscheck is not None:
            xgboost_path = staged / "xgboost_crosscheck.json"
            save_model = getattr(xgboost_crosscheck, "save_model", None)
            if not callable(save_model):
                raise ArtifactError(
                    "XGBoost cross-check does not expose native save_model"
                )
            save_model(str(xgboost_path))
            if not xgboost_path.is_file() or xgboost_path.stat().st_size == 0:
                raise ArtifactError("XGBoost did not emit a native model")
            _record(
                file_records,
                "xgboost_crosscheck",
                staged,
                xgboost_path,
                "xgboost-native-json",
            )

        if preprocessor_path is not None:
            source = Path(preprocessor_path).resolve()
            if not source.is_file():
                raise FileNotFoundError(source)
            target = staged / "preprocessor.json"
            # Validate JSON before copying bytes into the trusted bundle.
            with source.open("r", encoding="utf-8") as handle:
                json.load(handle)
            target.write_bytes(source.read_bytes())
            _record(file_records, "preprocessor", staged, target, "json")

        manifest = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "artifact_kind": "capacity-aware-aml-model-bundle",
            "bundle_id": identity,
            "created_at_utc": created_at,
            "primary_model": "lightgbm",
            "primary_serialization": "native-text",
            "feature_schema_version": schema["schema_version"],
            "files": file_records,
            "library_versions": _library_versions(
                (
                    "python",
                    "numpy",
                    "pandas",
                    "lightgbm",
                    "scikit-learn",
                    *(("xgboost",) if xgboost_crosscheck is not None else ()),
                )
            ),
        }
        _write_json(staged / MANIFEST_FILE, manifest)
        staged.replace(destination)
    return destination


def verify_model_bundle(path: str | Path) -> VerifiedBundle:
    """Validate manifest structure, paths, sizes, and every payload checksum."""
    root = Path(path).resolve()
    manifest_path = root / MANIFEST_FILE
    if not root.is_dir() or not manifest_path.is_file():
        raise ArtifactError(f"model bundle is missing {MANIFEST_FILE}: {root}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactError("model bundle manifest is unreadable") from error
    if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ArtifactError("unsupported model bundle schema version")
    if manifest.get("artifact_kind") != "capacity-aware-aml-model-bundle":
        raise ArtifactError("not a capacity-aware AML model bundle")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ArtifactError("model bundle has no file registry")
    for required in (
        "primary_model",
        "calibration",
        "feature_schema",
        "metrics",
        "training",
        "configuration",
    ):
        if required not in files:
            raise ArtifactError(f"model bundle is missing registry entry {required!r}")
    seen_paths: set[str] = set()
    for logical, record in files.items():
        if not isinstance(record, dict):
            raise ArtifactError(f"invalid file registry entry {logical!r}")
        relative = record.get("path")
        digest = record.get("sha256")
        size = record.get("bytes")
        if not isinstance(relative, str) or relative in seen_paths:
            raise ArtifactError(f"invalid or duplicate path for {logical!r}")
        seen_paths.add(relative)
        file_path = _safe_child(root, relative)
        if not file_path.is_file():
            raise ArtifactError(f"bundle payload is missing: {relative}")
        if int(size) != file_path.stat().st_size:
            raise ArtifactError(f"bundle payload size mismatch: {relative}")
        if not isinstance(digest, str) or sha256_file(file_path) != digest:
            raise ArtifactError(f"bundle payload checksum mismatch: {relative}")

    schema = _read_json(_safe_child(root, files["feature_schema"]["path"]))
    names = schema.get("feature_names")
    if (
        schema.get("schema_version") != manifest.get("feature_schema_version")
        or not isinstance(names, list)
        or not names
        or len(names) != len(set(names))
        or schema.get("feature_count") != len(names)
    ):
        raise ArtifactError("invalid feature schema")
    calibration = _read_json(_safe_child(root, files["calibration"]["path"]))
    try:
        PlattCalibrator.from_dict(calibration)
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactError("invalid calibration artifact") from error
    return VerifiedBundle(root, manifest)


def load_model_bundle(
    path: str | Path,
    *,
    allow_trusted_secondary_pickle: bool = False,
) -> LoadedModelBundle:
    """Verify and load a model bundle from a trusted local directory."""
    verified = verify_model_bundle(path)
    try:
        import lightgbm as lgb
    except ImportError as error:  # pragma: no cover - dependency path
        raise ModelDependencyError("LightGBM is required to load the model bundle") from error
    schema = _read_json(verified.path("feature_schema"))
    calibration = PlattCalibrator.from_dict(_read_json(verified.path("calibration")))
    booster = lgb.Booster(model_file=str(verified.path("primary_model")))
    names = tuple(str(item) for item in schema["feature_names"])
    booster_names = tuple(str(item) for item in booster.feature_name())
    if booster_names and booster_names != names:
        raise ArtifactError("native model features do not match feature_schema.json")
    model = LightGBMModel(booster, names, calibration)

    baseline = None
    files = verified.manifest["files"]
    if "logistic_baseline" in files:
        baseline = LinearLogitModel.from_dict(_read_json(verified.path("logistic_baseline")))
        if baseline.feature_names != names:
            raise ArtifactError("logistic baseline features do not match primary schema")

    anomaly = None
    if "anomaly_signal" in files and allow_trusted_secondary_pickle:
        try:
            import joblib
        except ImportError as error:  # pragma: no cover - dependency path
            raise ModelDependencyError("joblib is required to load Isolation Forest") from error
        payload = joblib.load(verified.path("anomaly_signal"))
        if not isinstance(payload, dict) or payload.get("kind") != "isolation-forest-signal":
            raise ArtifactError("invalid Isolation Forest artifact")
        anomaly_names = tuple(str(item) for item in payload["feature_names"])
        if anomaly_names != names:
            raise ArtifactError("Isolation Forest features do not match primary schema")
        scale = float(payload["scale"])
        if scale <= 0:
            raise ArtifactError("invalid Isolation Forest scale")
        anomaly = IsolationForestSignal(
            payload["estimator"], anomaly_names, float(payload["center"]), scale
        )

    return LoadedModelBundle(
        model=model,
        manifest=verified.manifest,
        feature_schema=schema,
        metrics=_read_json(verified.path("metrics")),
        training=_read_json(verified.path("training")),
        logistic_baseline=baseline,
        anomaly_signal=anomaly,
    )


def _record(
    records: dict[str, dict[str, Any]],
    logical: str,
    root: Path,
    path: Path,
    serialization: str,
) -> None:
    records[logical] = {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "serialization": serialization,
    }


def _safe_child(root: Path, relative: str) -> Path:
    candidate_relative = Path(relative)
    if candidate_relative.is_absolute() or ".." in candidate_relative.parts:
        raise ArtifactError(f"unsafe bundle path: {relative!r}")
    candidate = (root / candidate_relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise ArtifactError(f"bundle path escapes its root: {relative!r}") from error
    return candidate


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    try:
        rendered = json.dumps(
            value, indent=2, sort_keys=True, allow_nan=False, default=_json_default
        ) + "\n"
    except (TypeError, ValueError) as error:
        raise ArtifactError(f"artifact is not strict JSON: {path.name}") from error
    path.write_text(rendered, encoding="utf-8", newline="\n")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactError(f"invalid JSON artifact: {path.name}") from error
    if not isinstance(value, dict):
        raise ArtifactError(f"JSON artifact must be an object: {path.name}")
    return value


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _library_versions(names: Sequence[str]) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in names:
        if name == "python":
            import platform

            result[name] = platform.python_version()
            continue
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


__all__ = [
    "ArtifactError",
    "LoadedModelBundle",
    "VerifiedBundle",
    "load_model_bundle",
    "save_model_bundle",
    "sha256_file",
    "verify_model_bundle",
]
