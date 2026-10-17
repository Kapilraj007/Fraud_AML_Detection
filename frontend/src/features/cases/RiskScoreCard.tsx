import { Activity, BrainCircuit, CircleHelp, Radar } from "lucide-react";

import { formatPercent, riskTone } from "../../lib/format";

interface RiskScoreCardProps {
  combined: number;
  supervised?: number | null;
  anomaly?: number | null;
}

interface ScoreSignalProps {
  label: string;
  description: string;
  value?: number | null;
  icon: typeof Activity;
  prominent?: boolean;
}

export function RiskScoreCard({ combined, supervised, anomaly }: RiskScoreCardProps) {
  return (
    <section className="case-section risk-score-card" aria-labelledby="risk-score-heading">
      <header className="case-section__header">
        <span className="case-section__icon"><Activity aria-hidden="true" size={19} /></span>
        <div>
          <p className="case-section__eyebrow">Model evidence</p>
          <h3 id="risk-score-heading">Risk signals</h3>
        </div>
      </header>

      <div className="risk-score-card__signals">
        <ScoreSignal
          label="Combined risk"
          description="Calibrated supervised and anomaly signal"
          value={combined}
          icon={Activity}
          prominent
        />
        <ScoreSignal
          label="Supervised model"
          description="Calibrated fraud probability"
          value={supervised}
          icon={BrainCircuit}
        />
        <ScoreSignal
          label="Anomaly signal"
          description="Unusual-pattern probability"
          value={anomaly}
          icon={Radar}
        />
      </div>

      {(supervised == null || anomaly == null) ? (
        <p className="risk-score-card__availability">
          <CircleHelp aria-hidden="true" size={16} />
          Component scores are not exposed by the current case-detail response. The combined queue score remains authoritative.
        </p>
      ) : null}
    </section>
  );
}

function ScoreSignal({ label, description, value, icon: Icon, prominent }: ScoreSignalProps) {
  const bounded = value == null ? undefined : Math.min(1, Math.max(0, value));
  const tone = bounded == null ? "unavailable" : riskTone(bounded);

  return (
    <div className={`risk-signal${prominent ? " risk-signal--prominent" : ""}`}>
      <div className="risk-signal__header">
        <span className={`risk-signal__icon risk-signal__icon--${tone}`}>
          <Icon aria-hidden="true" size={18} />
        </span>
        <span>
          <strong>{label}</strong>
          <small>{description}</small>
        </span>
        <span className="risk-signal__value">{bounded == null ? "—" : formatPercent(bounded)}</span>
      </div>
      <div
        className={`risk-signal__meter risk-signal__meter--${tone}`}
        role={bounded == null ? undefined : "meter"}
        aria-label={label}
        aria-valuemin={bounded == null ? undefined : 0}
        aria-valuemax={bounded == null ? undefined : 100}
        aria-valuenow={bounded == null ? undefined : Math.round(bounded * 100)}
        aria-valuetext={bounded == null ? "Not available" : formatPercent(bounded)}
      >
        <span style={{ width: `${(bounded ?? 0) * 100}%` }} />
      </div>
    </div>
  );
}
