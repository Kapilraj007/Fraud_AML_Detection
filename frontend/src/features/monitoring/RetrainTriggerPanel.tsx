import { FormEvent, useId, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { CheckCircle2, LockKeyhole, Send, ShieldAlert } from "lucide-react";
import { PageState } from "../../components/ui/PageState";
import { api, normalizeApiError } from "../../lib/api";
import { formatDateTime, humanize } from "../../lib/format";
import type { DriftSnapshot, UserRole } from "../../lib/types";

type RetrainTriggerPanelProps = {
  snapshot: DriftSnapshot | null;
  snapshotLoading?: boolean;
  snapshotError?: string | null;
  onRetrySnapshot?: () => void;
  role: UserRole | undefined;
};

export function RetrainTriggerPanel({
  snapshot,
  snapshotLoading = false,
  snapshotError = null,
  onRetrySnapshot,
  role,
}: RetrainTriggerPanelProps) {
  const reasonId = useId();
  const [reason, setReason] = useState("");
  const canRequest = role === "senior_investigator" || role === "admin";
  const mutation = useMutation({
    mutationFn: (requestReason: string) => {
      if (!snapshot) throw new Error("A monitoring snapshot is required.");
      return api.monitoring.requestRetraining({
        drift_snapshot_id: snapshot.id,
        reason: requestReason.trim(),
      });
    },
    onSuccess: () => setReason(""),
  });

  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!canRequest || !snapshot || reason.trim().length < 8) return;
    mutation.mutate(reason);
  };

  return (
    <section className="panel retraining-panel" aria-labelledby="retraining-title">
      <div className="panel-header">
        <div>
          <p className="panel-eyebrow">Human-gated control</p>
          <h3 id="retraining-title">Request model retraining</h3>
          <p>
            Open a governed request tied to a specific drift snapshot. This does not retrain or deploy a model.
          </p>
        </div>
        <ShieldAlert size={21} aria-hidden="true" />
      </div>

      {!snapshot && snapshotLoading ? (
        <PageState
          kind="loading"
          title="Loading active-model snapshot"
          message="Retraining controls will be available after the current model evidence is resolved."
        />
      ) : !snapshot && snapshotError ? (
        <PageState
          kind="error"
          title="Active-model snapshot unavailable"
          message={snapshotError}
          onRetry={onRetrySnapshot}
        />
      ) : !snapshot ? (
        <PageState
          kind="empty"
          title="No active-model snapshot"
          message="The active model has no persisted monitoring window. A retraining request cannot be opened yet."
        />
      ) : !canRequest ? (
        <div className="permission-state">
          <span className="permission-state-icon" aria-hidden="true"><LockKeyhole size={20} /></span>
          <div>
            <strong>Senior approval required</strong>
            <p>
              Your role can inspect monitoring evidence. Senior investigators and administrators can
              submit retraining requests.
            </p>
          </div>
        </div>
      ) : (
        <form className="retraining-form" onSubmit={submit}>
          <div className="snapshot-reference">
            <span>Request evidence</span>
            <strong>{humanize(snapshot.status)} · {formatDateTime(snapshot.window_end)}</strong>
            <small>Snapshot {snapshot.id}</small>
          </div>

          <div className="form-field">
            <label htmlFor={reasonId}>Reason for retraining review</label>
            <textarea
              id={reasonId}
              value={reason}
              minLength={8}
              maxLength={2000}
              rows={5}
              required
              placeholder="Describe the sustained signal, affected features, and why a new training window should be evaluated."
              onChange={(event) => {
                setReason(event.target.value);
                if (mutation.isSuccess || mutation.isError) mutation.reset();
              }}
              aria-describedby={`${reasonId}-hint`}
            />
            <div className="form-field-hint" id={`${reasonId}-hint`}>
              <span>Minimum 8 characters</span>
              <span>{reason.length}/2000</span>
            </div>
          </div>

          {mutation.isError ? (
            <div className="inline-alert inline-alert-error" role="alert">
              {normalizeApiError(mutation.error, "The retraining request could not be created.")}
            </div>
          ) : null}
          {mutation.isSuccess ? (
            <div className="inline-alert inline-alert-success" role="status">
              <CheckCircle2 size={18} aria-hidden="true" />
              Request {mutation.data.id} opened with status {humanize(mutation.data.status)}.
            </div>
          ) : null}

          <div className="form-actions">
            <p>Submission creates an auditable ticket for human review.</p>
            <button
              className="button button-primary"
              type="submit"
              disabled={mutation.isPending || reason.trim().length < 8}
            >
              <Send size={16} aria-hidden="true" />
              {mutation.isPending ? "Submitting…" : "Submit request"}
            </button>
          </div>
        </form>
      )}
    </section>
  );
}
