import {
  ArrowDownRight,
  ArrowUpRight,
  Ban,
  CheckCircle2,
  CircleHelp,
  Sparkles,
} from "lucide-react";

import type { Explanation } from "../../lib/types";

interface ExplanationPanelProps {
  explanation?: Explanation;
  reliabilityGated: boolean;
}

interface FeatureContribution {
  feature?: string;
  display_name?: string;
  value?: number;
  direction?: string;
}

export function ExplanationPanel({ explanation, reliabilityGated }: ExplanationPanelProps) {
  if (!explanation) {
    return (
      <section className="case-section explanation-panel" aria-labelledby="explanation-heading">
        <ExplanationHeader />
        <div className="case-empty-state">
          <CircleHelp aria-hidden="true" size={21} />
          <p>The current case-detail response does not include feature-level explanation evidence.</p>
        </div>
      </section>
    );
  }

  const explanationIsReliable = explanation.reliable ?? !explanation.is_withheld;
  const reliabilityReasons = explanation.reliability_reasons?.length
    ? explanation.reliability_reasons
    : explanation.gate_reason
      ? [explanation.gate_reason]
      : [];

  if (reliabilityGated && !explanationIsReliable) {
    return (
      <section className="case-section explanation-panel" aria-labelledby="explanation-heading">
        <ExplanationHeader />
        <div className="explanation-panel__withheld" role="note">
          <Ban aria-hidden="true" size={22} />
          <div>
            <strong>Explanation withheld</strong>
            <p>
              Reliability checks did not pass, so feature contributions are hidden for this review.
            </p>
            {reliabilityReasons.length > 0 ? (
              <ul>
                {reliabilityReasons.map((reason) => <li key={reason}>{reason}</li>)}
              </ul>
            ) : null}
          </div>
        </div>
      </section>
    );
  }

  const contributions = normalizeFeatures(explanation.top_features).slice(0, 5);
  const maxMagnitude = Math.max(1e-12, ...contributions.map((item) => Math.abs(item.value)));

  return (
    <section className="case-section explanation-panel" aria-labelledby="explanation-heading">
      <ExplanationHeader />

      {reliabilityGated ? (
        <div className="explanation-panel__reliability explanation-panel__reliability--passed">
          <CheckCircle2 aria-hidden="true" size={17} />
          Reliability checks passed
        </div>
      ) : null}

      {contributions.length === 0 ? (
        <div className="case-empty-state">
          <CircleHelp aria-hidden="true" size={21} />
          <p>No feature contributions are available for this prediction.</p>
        </div>
      ) : (
        <figure className="explanation-chart">
          <figcaption className="sr-only">Top signed feature contributions to risk</figcaption>
          <div className="explanation-chart__legend" aria-hidden="true">
            <span>Decreases risk</span>
            <span>Increases risk</span>
          </div>
          <ol className="explanation-chart__list">
            {contributions.map((item, index) => (
              <li className="explanation-bar" key={`${item.feature}-${index}`}>
                <div className="explanation-bar__label">
                  <strong>{item.label}</strong>
                  <span className={item.value >= 0 ? "is-positive" : "is-negative"}>
                    {item.value >= 0 ? <ArrowUpRight aria-hidden="true" size={15} /> : <ArrowDownRight aria-hidden="true" size={15} />}
                    {item.value >= 0 ? "Raises" : "Lowers"} risk · {formatSigned(item.value)}
                  </span>
                </div>
                <div
                  className="explanation-bar__track"
                  role="meter"
                  aria-label={`${item.label} ${item.value >= 0 ? "increases" : "decreases"} risk by ${Math.abs(item.value).toFixed(3)}`}
                  aria-valuemin={-maxMagnitude}
                  aria-valuemax={maxMagnitude}
                  aria-valuenow={item.value}
                >
                  <span className="explanation-bar__axis" aria-hidden="true" />
                  <span
                    className={`explanation-bar__value ${item.value >= 0 ? "explanation-bar__value--positive" : "explanation-bar__value--negative"}`}
                    style={{ width: `${(Math.abs(item.value) / maxMagnitude) * 50}%` }}
                    aria-hidden="true"
                  />
                </div>
              </li>
            ))}
          </ol>
        </figure>
      )}

      {explanation.narrative ? (
        <div className="explanation-panel__narrative">
          <strong>Analyst summary</strong>
          <p>{explanation.narrative}</p>
        </div>
      ) : null}
    </section>
  );
}

function ExplanationHeader() {
  return (
    <header className="case-section__header">
      <span className="case-section__icon"><Sparkles aria-hidden="true" size={19} /></span>
      <div>
        <p className="case-section__eyebrow">Decision support</p>
        <h3 id="explanation-heading">Why this case was flagged</h3>
      </div>
    </header>
  );
}

function normalizeFeatures(features: Explanation["top_features"]): Array<{
  feature: string;
  label: string;
  value: number;
}> {
  return (features as FeatureContribution[])
    .map((feature, index) => ({
      feature: feature.feature ?? `feature-${index}`,
      label: feature.display_name ?? feature.feature?.replaceAll("_", " ") ?? `Factor ${index + 1}`,
      value: Number(feature.value),
    }))
    .filter((feature) => Number.isFinite(feature.value))
    .sort((left, right) => Math.abs(right.value) - Math.abs(left.value));
}

function formatSigned(value: number) {
  return `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(3)}`;
}
