"""Human-readable explanations and reliability gating for analyst review."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


DISPLAY_NAMES = {
    "amount_log1p": "transaction amount",
    "amount": "transaction amount",
    "source_txn_count_1h": "one-hour transaction velocity",
    "source_txn_count_24h": "24-hour transaction velocity",
    "source_txn_sum_24h": "24-hour transferred value",
    "amount_to_source_mean_30d_ratio": "amount relative to account history",
    "is_new_counterparty": "new counterparty activity",
    "new_device_flag": "a previously unseen device",
    "source_fan_out_24h": "outbound fan-out",
    "destination_fan_in_24h": "inbound fan-in",
    "rapid_pass_through_ratio": "rapid pass-through activity",
    "causal_chain_depth": "transaction-chain depth",
}


@dataclass(frozen=True)
class FeatureContribution:
    feature: str
    value: float
    direction: str
    display_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "display_name": self.display_name,
            "value": self.value,
            "direction": self.direction,
        }


@dataclass(frozen=True)
class ExplanationResult:
    top_features: tuple[FeatureContribution, ...]
    narrative: str
    reliable: bool
    reliability_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "top_features": [item.to_dict() for item in self.top_features],
            "narrative": self.narrative,
            "reliable": self.reliable,
            "reliability_reasons": list(self.reliability_reasons),
        }


def explain_contributions(
    contributions: Mapping[str, float],
    *,
    top_k: int = 5,
    model_probability: float | None = None,
    anomaly_probability: float | None = None,
    drift_detected: bool = False,
    minimum_total_magnitude: float = 1e-8,
    disagreement_limit: float = 0.65,
) -> ExplanationResult:
    """Rank signed TreeSHAP values and attach explicit reliability reasons."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    clean: list[tuple[str, float]] = []
    for name, raw in contributions.items():
        if name == "expected_value":
            continue
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f"non-finite contribution for {name}")
        clean.append((str(name), value))
    clean.sort(key=lambda item: (-abs(item[1]), item[0]))
    top = tuple(
        FeatureContribution(
            feature=name,
            value=value,
            direction="increases_risk" if value >= 0 else "decreases_risk",
            display_name=_display_name(name),
        )
        for name, value in clean[:top_k]
    )

    reasons: list[str] = []
    magnitude = sum(abs(value) for _, value in clean)
    if magnitude < minimum_total_magnitude:
        reasons.append("model contributions are too small to support a stable rationale")
    if drift_detected:
        reasons.append("current inputs are under a detected distribution shift")
    if model_probability is not None and anomaly_probability is not None:
        if abs(float(model_probability) - float(anomaly_probability)) > disagreement_limit:
            reasons.append("supervised and anomaly signals strongly disagree")

    if not top:
        narrative = "No feature-level rationale is available for this prediction."
    else:
        fragments = []
        for item in top[:3]:
            verb = "higher" if item.value >= 0 else "lower"
            fragments.append(f"{item.display_name} ({verb} risk)")
        narrative = "Risk is driven primarily by " + ", ".join(fragments) + "."
    if reasons:
        narrative += " Treat this explanation as low reliability: " + "; ".join(reasons) + "."
    return ExplanationResult(top, narrative, not reasons, tuple(reasons))


def _display_name(name: str) -> str:
    if name in DISPLAY_NAMES:
        return DISPLAY_NAMES[name]
    cleaned = name.replace("__", " ").replace("_", " ").replace("=", " ")
    return " ".join(cleaned.split())


__all__ = [
    "ExplanationResult",
    "FeatureContribution",
    "explain_contributions",
]
