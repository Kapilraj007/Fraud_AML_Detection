"""Model primitives for Sprint 2 AML risk scoring.

The primary classifier is a native LightGBM ``Booster``.  LightGBM and
scikit-learn are imported lazily so schema validation, artifact verification,
and metric tests remain usable in minimal environments.  The primary model is
never pickled: :mod:`backend.app.ml.artifacts` persists it with
``Booster.save_model``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


class ModelDependencyError(RuntimeError):
    """Raised when an optional model dependency is not installed."""


class ModelInputError(ValueError):
    """Raised when an inference matrix violates the frozen feature contract."""


def stable_sigmoid(values: Any) -> np.ndarray:
    """Return a numerically stable logistic transform."""
    array = np.asarray(values, dtype=np.float64)
    result = np.empty_like(array, dtype=np.float64)
    positive = array >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    exponential = np.exp(array[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def validate_binary_labels(
    labels: Sequence[Any] | pd.Series | np.ndarray,
    *,
    name: str = "labels",
    minimum_positives: int = 1,
    minimum_negatives: int = 1,
) -> np.ndarray:
    """Validate a complete binary target and return compact integer labels."""
    values = pd.to_numeric(pd.Series(labels), errors="coerce").to_numpy(dtype=float)
    if values.ndim != 1 or not len(values):
        raise ModelInputError(f"{name} must contain at least one row")
    if not np.isfinite(values).all():
        raise ModelInputError(f"{name} contains missing or non-finite values")
    unique = set(np.unique(values).tolist())
    if not unique <= {0.0, 1.0}:
        raise ModelInputError(f"{name} must be binary; observed {sorted(unique)}")
    result = values.astype(np.int8)
    positives = int(result.sum())
    negatives = int(len(result) - positives)
    if positives < minimum_positives or negatives < minimum_negatives:
        raise ModelInputError(
            f"{name} has positives={positives}, negatives={negatives}; requires "
            f"at least {minimum_positives} positives and {minimum_negatives} negatives"
        )
    return result


def validated_matrix(
    values: pd.DataFrame | np.ndarray,
    feature_names: Sequence[str],
) -> np.ndarray:
    """Validate and order a matrix against an exact model schema."""
    names = tuple(str(name) for name in feature_names)
    if not names or len(names) != len(set(names)):
        raise ModelInputError("feature_names must be non-empty and unique")
    if isinstance(values, pd.DataFrame):
        missing = [name for name in names if name not in values.columns]
        if missing:
            raise ModelInputError(f"model inputs are missing features: {missing}")
        matrix = values.loc[:, list(names)].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=np.float64, copy=False)
    else:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2 or matrix.shape[1] != len(names):
            raise ModelInputError(
                f"model matrix must have {len(names)} columns, got {matrix.shape}"
            )
    if not np.isfinite(matrix).all():
        raise ModelInputError("model inputs contain missing or non-finite values")
    return matrix


@dataclass(frozen=True)
class PlattCalibrator:
    """A monotone sigmoid calibrator fitted to validation-set raw margins."""

    slope: float = 1.0
    intercept: float = 0.0
    fitted_rows: int = 0
    positive_rows: int = 0
    version: int = 1

    @classmethod
    def fit(
        cls,
        raw_scores: Sequence[float] | np.ndarray,
        labels: Sequence[Any] | np.ndarray,
        *,
        sample_weight: Sequence[float] | np.ndarray | None = None,
        maximum_iterations: int = 100,
        tolerance: float = 1e-9,
        l2: float = 1e-8,
    ) -> "PlattCalibrator":
        """Fit non-negative-slope logistic calibration with damped Newton steps."""
        scores = np.asarray(raw_scores, dtype=np.float64).reshape(-1)
        target = validate_binary_labels(labels, name="calibration labels")
        if len(scores) != len(target) or not np.isfinite(scores).all():
            raise ModelInputError("calibration scores must be finite and match labels")
        if sample_weight is None:
            weights = np.ones(len(scores), dtype=np.float64)
        else:
            weights = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
            if len(weights) != len(scores) or not np.isfinite(weights).all():
                raise ModelInputError("calibration weights must be finite and match rows")
            if np.any(weights <= 0):
                raise ModelInputError("calibration weights must be positive")

        prevalence = float(np.average(target, weights=weights))
        prevalence = min(max(prevalence, 1e-9), 1.0 - 1e-9)
        parameters = np.array([1.0, math.log(prevalence / (1.0 - prevalence))])

        def loss(candidate: np.ndarray) -> float:
            logits = candidate[0] * scores + candidate[1]
            # logaddexp is the stable binary cross-entropy primitive.
            return float(np.sum(weights * (np.logaddexp(0.0, logits) - target * logits))
                         + 0.5 * l2 * candidate[0] ** 2)

        current_loss = loss(parameters)
        design = np.column_stack((scores, np.ones(len(scores))))
        for _ in range(int(maximum_iterations)):
            probabilities = stable_sigmoid(design @ parameters)
            gradient = design.T @ (weights * (probabilities - target))
            gradient[0] += l2 * parameters[0]
            curvature = weights * probabilities * (1.0 - probabilities)
            hessian = design.T @ (design * curvature[:, None])
            hessian[0, 0] += l2
            hessian.flat[::3] += 1e-12
            try:
                step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                step = np.linalg.pinv(hessian) @ gradient
            if np.linalg.norm(step, ord=np.inf) <= tolerance:
                break

            accepted = False
            damping = 1.0
            for _ in range(30):
                candidate = parameters - damping * step
                candidate[0] = max(0.0, candidate[0])
                candidate_loss = loss(candidate)
                if math.isfinite(candidate_loss) and candidate_loss <= current_loss:
                    parameters, current_loss = candidate, candidate_loss
                    accepted = True
                    break
                damping *= 0.5
            if not accepted:
                break

        if not np.isfinite(parameters).all():
            raise RuntimeError("calibration did not converge to finite parameters")
        return cls(
            slope=float(parameters[0]),
            intercept=float(parameters[1]),
            fitted_rows=len(target),
            positive_rows=int(target.sum()),
        )

    def predict(self, raw_scores: Sequence[float] | np.ndarray) -> np.ndarray:
        scores = np.asarray(raw_scores, dtype=np.float64)
        if not np.isfinite(scores).all():
            raise ModelInputError("calibration scores contain non-finite values")
        return stable_sigmoid(self.slope * scores + self.intercept)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "platt-sigmoid",
            "version": self.version,
            "slope": self.slope,
            "intercept": self.intercept,
            "fitted_rows": self.fitted_rows,
            "positive_rows": self.positive_rows,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PlattCalibrator":
        if value.get("kind") != "platt-sigmoid" or int(value.get("version", -1)) != 1:
            raise ValueError("unsupported calibration artifact")
        slope, intercept = float(value["slope"]), float(value["intercept"])
        if not math.isfinite(slope) or slope < 0 or not math.isfinite(intercept):
            raise ValueError("invalid calibration coefficients")
        rows, positives = int(value.get("fitted_rows", 0)), int(value.get("positive_rows", 0))
        if rows < 0 or positives < 0 or positives > rows:
            raise ValueError("invalid calibration row counts")
        return cls(slope, intercept, rows, positives)


@dataclass
class LightGBMModel:
    """Schema-enforcing wrapper around a native LightGBM Booster."""

    booster: Any
    feature_names: tuple[str, ...]
    calibrator: PlattCalibrator | None = None

    def __post_init__(self) -> None:
        self.feature_names = tuple(str(name) for name in self.feature_names)
        if not self.feature_names or len(self.feature_names) != len(set(self.feature_names)):
            raise ModelInputError("LightGBM feature names must be non-empty and unique")

    @property
    def best_iteration(self) -> int | None:
        value = getattr(self.booster, "best_iteration", None)
        return int(value) if value else None

    def predict_margin(self, values: pd.DataFrame | np.ndarray) -> np.ndarray:
        matrix = validated_matrix(values, self.feature_names)
        predictions = self.booster.predict(
            matrix, raw_score=True, num_iteration=self.best_iteration
        )
        answer = np.asarray(predictions, dtype=np.float64).reshape(-1)
        if not np.isfinite(answer).all():
            raise RuntimeError("LightGBM returned non-finite raw scores")
        return answer

    def predict_probability(self, values: pd.DataFrame | np.ndarray) -> np.ndarray:
        margin = self.predict_margin(values)
        return (self.calibrator.predict(margin) if self.calibrator is not None
                else stable_sigmoid(margin))

    def predict_contributions(self, values: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Return TreeSHAP contributions plus the expected-value column."""
        matrix = validated_matrix(values, self.feature_names)
        result = self.booster.predict(
            matrix, pred_contrib=True, num_iteration=self.best_iteration
        )
        contributions = np.asarray(result, dtype=np.float64)
        expected_columns = len(self.feature_names) + 1
        if contributions.ndim != 2 or contributions.shape[1] != expected_columns:
            raise RuntimeError(
                "LightGBM contribution output does not match the feature schema"
            )
        columns = [*self.feature_names, "expected_value"]
        return pd.DataFrame(contributions, columns=columns)


@dataclass(frozen=True)
class LinearLogitModel:
    """JSON-serializable logistic-regression baseline."""

    feature_names: tuple[str, ...]
    coefficients: tuple[float, ...]
    intercept: float

    def predict_margin(self, values: pd.DataFrame | np.ndarray) -> np.ndarray:
        matrix = validated_matrix(values, self.feature_names)
        return matrix @ np.asarray(self.coefficients, dtype=float) + self.intercept

    def predict_probability(self, values: pd.DataFrame | np.ndarray) -> np.ndarray:
        return stable_sigmoid(self.predict_margin(values))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "linear-logit-baseline",
            "version": 1,
            "feature_names": list(self.feature_names),
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LinearLogitModel":
        if value.get("kind") != "linear-logit-baseline" or value.get("version") != 1:
            raise ValueError("unsupported logistic baseline artifact")
        names = tuple(str(item) for item in value["feature_names"])
        coefficients = tuple(float(item) for item in value["coefficients"])
        intercept = float(value["intercept"])
        if len(names) != len(coefficients) or not names:
            raise ValueError("invalid logistic baseline schema")
        if not np.isfinite([*coefficients, intercept]).all():
            raise ValueError("non-finite logistic baseline coefficients")
        return cls(names, coefficients, intercept)


@dataclass
class IsolationForestSignal:
    """Normalized high-is-anomalous score from a fitted Isolation Forest."""

    estimator: Any
    feature_names: tuple[str, ...]
    center: float
    scale: float

    def predict(self, values: pd.DataFrame | np.ndarray) -> np.ndarray:
        matrix = validated_matrix(values, self.feature_names)
        raw = -np.asarray(self.estimator.score_samples(matrix), dtype=float)
        return stable_sigmoid((raw - self.center) / self.scale)


def train_lightgbm(
    train_features: pd.DataFrame,
    train_labels: Sequence[Any],
    validation_features: pd.DataFrame,
    validation_labels: Sequence[Any],
    *,
    parameters: Mapping[str, Any] | None = None,
    sample_weight: Sequence[float] | np.ndarray | None = None,
) -> LightGBMModel:
    """Train the primary CPU LightGBM model with validation early stopping."""
    try:
        import lightgbm as lgb
    except ImportError as error:  # pragma: no cover - dependency path
        raise ModelDependencyError(
            "LightGBM is required for training; install the pinned project dependencies"
        ) from error

    if not isinstance(train_features, pd.DataFrame) or not isinstance(
        validation_features, pd.DataFrame
    ):
        raise ModelInputError("LightGBM training inputs must be pandas DataFrames")
    feature_names = tuple(str(name) for name in train_features.columns)
    if tuple(validation_features.columns) != feature_names:
        raise ModelInputError("training and validation feature schemas differ")
    train_matrix = validated_matrix(train_features, feature_names)
    validation_matrix = validated_matrix(validation_features, feature_names)
    train_target = validate_binary_labels(train_labels, name="training labels")
    validation_target = validate_binary_labels(
        validation_labels, name="validation labels"
    )
    if len(train_matrix) != len(train_target) or len(validation_matrix) != len(validation_target):
        raise ModelInputError("feature and label row counts differ")

    options = dict(parameters or {})
    num_boost_round = int(options.pop("num_boost_round", 2000))
    early_stopping_rounds = int(options.pop("early_stopping_rounds", 100))
    log_evaluation_period = int(options.pop("log_evaluation_period", 0))
    categorical_features = options.pop("categorical_feature", "auto")
    defaults = {
        "objective": "binary",
        "metric": ["average_precision", "auc"],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "max_bin": 63,
        "verbosity": -1,
        "seed": 20261010,
        "feature_fraction_seed": 20261010,
        "bagging_seed": 20261010,
        "data_random_seed": 20261010,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 4,
    }
    defaults.update(options)
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    if weights is not None and (len(weights) != len(train_target)
                                or not np.isfinite(weights).all()
                                or np.any(weights <= 0)):
        raise ModelInputError("training sample weights are invalid")
    train_set = lgb.Dataset(
        train_matrix,
        label=train_target,
        weight=weights,
        feature_name=list(feature_names),
        categorical_feature=categorical_features,
        free_raw_data=True,
    )
    validation_set = lgb.Dataset(
        validation_matrix,
        label=validation_target,
        feature_name=list(feature_names),
        categorical_feature=categorical_features,
        reference=train_set,
        free_raw_data=True,
    )
    callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=False)]
    if log_evaluation_period > 0:
        callbacks.append(lgb.log_evaluation(log_evaluation_period))
    booster = lgb.train(
        defaults,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[validation_set],
        valid_names=["validation"],
        callbacks=callbacks,
    )
    return LightGBMModel(booster, feature_names)


def train_logistic_baseline(
    train_features: pd.DataFrame,
    train_labels: Sequence[Any],
    *,
    sample_weight: Sequence[float] | np.ndarray | None = None,
    class_weight: str | Mapping[int, float] | None = "balanced",
    seed: int = 20261010,
) -> LinearLogitModel:
    """Fit an L2 logistic baseline and return a portable JSON representation."""
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as error:  # pragma: no cover - dependency path
        raise ModelDependencyError("scikit-learn is required for the baseline") from error
    names = tuple(str(name) for name in train_features.columns)
    matrix = validated_matrix(train_features, names)
    target = validate_binary_labels(train_labels, name="training labels")
    estimator = LogisticRegression(
        class_weight=class_weight,
        max_iter=1000,
        random_state=seed,
        solver="lbfgs",
    )
    estimator.fit(matrix, target, sample_weight=sample_weight)
    return LinearLogitModel(
        names,
        tuple(float(value) for value in estimator.coef_.reshape(-1)),
        float(estimator.intercept_[0]),
    )


def train_isolation_forest(
    train_features: pd.DataFrame,
    *,
    number_of_estimators: int = 200,
    maximum_samples: int = 8192,
    seed: int = 20261010,
    jobs: int = 4,
) -> IsolationForestSignal:
    """Fit a deterministic complementary anomaly signal on training inputs only."""
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError as error:  # pragma: no cover - dependency path
        raise ModelDependencyError("scikit-learn is required for Isolation Forest") from error
    names = tuple(str(name) for name in train_features.columns)
    matrix = validated_matrix(train_features, names)
    estimator = IsolationForest(
        n_estimators=int(number_of_estimators),
        max_samples=min(int(maximum_samples), len(matrix)),
        contamination="auto",
        random_state=int(seed),
        n_jobs=int(jobs),
    )
    estimator.fit(matrix)
    raw = -np.asarray(estimator.score_samples(matrix), dtype=float)
    center = float(np.median(raw))
    mad = float(np.median(np.abs(raw - center)))
    scale = max(1.4826 * mad, 1e-9)
    return IsolationForestSignal(estimator, names, center, scale)


def train_xgboost_crosscheck(
    train_features: pd.DataFrame,
    train_labels: Sequence[Any],
    validation_features: pd.DataFrame,
    validation_labels: Sequence[Any],
    *,
    parameters: Mapping[str, Any] | None = None,
    sample_weight: Sequence[float] | np.ndarray | None = None,
) -> Any:
    """Optionally fit an offline XGBoost robustness cross-check."""
    try:
        import xgboost as xgb
    except ImportError as error:  # pragma: no cover - optional path
        raise ModelDependencyError(
            "XGBoost cross-check requested but xgboost is not installed"
        ) from error
    names = tuple(str(name) for name in train_features.columns)
    train_matrix = validated_matrix(train_features, names)
    validation_matrix = validated_matrix(validation_features, names)
    train_target = validate_binary_labels(train_labels, name="training labels")
    validation_target = validate_binary_labels(validation_labels, name="validation labels")
    options = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 20261010,
        "nthread": 4,
    }
    options.update(parameters or {})
    rounds = int(options.pop("num_boost_round", 1000))
    early = int(options.pop("early_stopping_rounds", 75))
    train_dmatrix = xgb.DMatrix(
        train_matrix, label=train_target, weight=sample_weight, feature_names=list(names)
    )
    validation_dmatrix = xgb.DMatrix(
        validation_matrix, label=validation_target, feature_names=list(names)
    )
    return xgb.train(
        options,
        train_dmatrix,
        num_boost_round=rounds,
        evals=[(validation_dmatrix, "validation")],
        early_stopping_rounds=early,
        verbose_eval=False,
    )


__all__ = [
    "IsolationForestSignal",
    "LightGBMModel",
    "LinearLogitModel",
    "ModelDependencyError",
    "ModelInputError",
    "PlattCalibrator",
    "stable_sigmoid",
    "train_isolation_forest",
    "train_lightgbm",
    "train_logistic_baseline",
    "train_xgboost_crosscheck",
    "validate_binary_labels",
    "validated_matrix",
]
