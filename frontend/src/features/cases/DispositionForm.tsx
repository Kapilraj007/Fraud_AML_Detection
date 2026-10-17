import { AlertTriangle, CheckCircle2, HelpCircle, Send, XCircle } from "lucide-react";
import { useId, useMemo, useState } from "react";

import { normalizeApiError } from "../../lib/api";
import type { CaseDispositionCreate, DispositionAction } from "../../lib/types";

interface DispositionFormProps {
  elapsedSeconds: number;
  isSubmitting: boolean;
  onSubmit: (payload: CaseDispositionCreate) => Promise<void>;
}

const DECISIONS: Array<{
  value: DispositionAction;
  label: string;
  description: string;
  icon: typeof AlertTriangle;
}> = [
  {
    value: "escalate",
    label: "Escalate",
    description: "Send for senior investigation",
    icon: AlertTriangle,
  },
  {
    value: "dismiss",
    label: "Dismiss",
    description: "Close with no further action",
    icon: XCircle,
  },
  {
    value: "request_more_info",
    label: "Request more info",
    description: "Pause pending more evidence",
    icon: HelpCircle,
  },
];

export function DispositionForm({ elapsedSeconds, isSubmitting, onSubmit }: DispositionFormProps) {
  const formId = useId();
  const [decision, setDecision] = useState<DispositionAction | "">("");
  const [confidence, setConfidence] = useState(3);
  const [rationale, setRationale] = useState("");
  const [modelAgreement, setModelAgreement] = useState<"agree" | "disagree" | "unsure">("unsure");
  const [error, setError] = useState<string>();
  const safeElapsed = Math.min(86_400, Math.max(0, Math.floor(elapsedSeconds)));
  const elapsedLabel = useMemo(() => formatDuration(safeElapsed), [safeElapsed]);

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(undefined);
    if (!decision) {
      setError("Select a disposition before submitting.");
      return;
    }
    if (!rationale.trim()) {
      setError("Add a concise rationale for the audit record.");
      return;
    }

    try {
      await onSubmit({
        decision,
        confidence,
        rationale: rationale.trim(),
        model_agreement: modelAgreement === "unsure" ? null : modelAgreement === "agree",
        review_duration_seconds: safeElapsed,
      });
    } catch (caught) {
      setError(normalizeApiError(caught));
    }
  }

  return (
    <section className="case-section disposition" aria-labelledby={`${formId}-heading`}>
      <header className="case-section__header">
        <span className="case-section__icon"><CheckCircle2 aria-hidden="true" size={19} /></span>
        <div>
          <p className="case-section__eyebrow">Required decision</p>
          <h3 id={`${formId}-heading`}>Record disposition</h3>
        </div>
        <span className="disposition__timer" aria-label={`Review time ${elapsedLabel}`}>
          {elapsedLabel}
        </span>
      </header>

      <form className="disposition__form" onSubmit={handleSubmit}>
        <fieldset className="disposition__decisions">
          <legend>Decision</legend>
          <div className="disposition__decision-grid">
            {DECISIONS.map((item) => {
              const Icon = item.icon;
              return (
                <label className={`decision-option decision-option--${item.value}`} key={item.value}>
                  <input
                    type="radio"
                    name={`${formId}-decision`}
                    value={item.value}
                    checked={decision === item.value}
                    onChange={() => setDecision(item.value)}
                  />
                  <Icon aria-hidden="true" size={19} />
                  <span><strong>{item.label}</strong><small>{item.description}</small></span>
                </label>
              );
            })}
          </div>
        </fieldset>

        <fieldset className="disposition__confidence">
          <legend>Decision confidence</legend>
          <p>How confident are you in this assessment?</p>
          <div className="confidence-scale">
            {[1, 2, 3, 4, 5].map((value) => (
              <label key={value}>
                <input
                  type="radio"
                  name={`${formId}-confidence`}
                  value={value}
                  checked={confidence === value}
                  onChange={() => setConfidence(value)}
                />
                <span>{value}</span>
              </label>
            ))}
          </div>
          <div className="confidence-scale__labels" aria-hidden="true"><span>Low</span><span>High</span></div>
        </fieldset>

        <fieldset className="disposition__agreement">
          <legend>Model agreement</legend>
          <div className="segmented-control">
            {[
              ["agree", "Agree"],
              ["disagree", "Disagree"],
              ["unsure", "Not sure"],
            ].map(([value, label]) => (
              <label key={value}>
                <input
                  type="radio"
                  name={`${formId}-agreement`}
                  value={value}
                  checked={modelAgreement === value}
                  onChange={() => setModelAgreement(value as typeof modelAgreement)}
                />
                <span>{label}</span>
              </label>
            ))}
          </div>
        </fieldset>

        <label className="disposition__rationale">
          <span>Rationale <strong aria-hidden="true">*</strong></span>
          <textarea
            value={rationale}
            maxLength={4000}
            rows={5}
            required
            placeholder="Summarize the evidence that supports your decision…"
            onChange={(event) => setRationale(event.target.value)}
          />
          <small>{rationale.length.toLocaleString()} / 4,000</small>
        </label>

        {error ? <div className="form-error" role="alert">{error}</div> : null}

        <div className="disposition__footer">
          <p>This action is recorded in the immutable audit history.</p>
          <button className="button button--primary" type="submit" disabled={isSubmitting}>
            <Send aria-hidden="true" size={17} />
            {isSubmitting ? "Recording…" : "Record disposition"}
          </button>
        </div>
      </form>
    </section>
  );
}

function formatDuration(totalSeconds: number) {
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  return hours > 0
    ? `${hours}:${minutes.toString().padStart(2, "0")}:${seconds.toString().padStart(2, "0")}`
    : `${minutes}:${seconds.toString().padStart(2, "0")}`;
}
