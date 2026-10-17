import { useQuery } from "@tanstack/react-query";
import {
  Activity,
  ArrowRight,
  BriefcaseBusiness,
  CircleAlert,
  Clock3,
  Database,
  Gauge,
  ScanSearch,
  ShieldCheck,
} from "lucide-react";
import { Link } from "react-router-dom";
import { useAuth } from "../../auth/AuthContext";
import { PageState } from "../../components/ui/PageState";
import { api } from "../../lib/api";
import {
  caseStatusLabel,
  formatCompactNumber,
  formatDateTime,
  formatPercent,
  formatRelativeTime,
  humanize,
  riskTone,
} from "../../lib/format";
import type { Case, DriftStatus } from "../../lib/types";
import { getStudyArmPresentation } from "../cases/studyArms";
import "./operations-shared.css";
import "./overview.css";

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "The service returned an unexpected response.";
}

function driftTone(status: DriftStatus | null | undefined) {
  if (status === "drift_detected") return "critical";
  if (status === "watch") return "warning";
  if (status === "stable") return "success";
  return "neutral";
}

function caseTone(caseItem: Case) {
  if (caseItem.status === "escalated") return "critical";
  if (caseItem.status === "dismissed" || caseItem.status === "closed") return "success";
  if (caseItem.status === "in_review" || caseItem.status === "claimed") return "warning";
  return "neutral";
}

export function OverviewPage() {
  const { user } = useAuth();
  const casesQuery = useQuery({
    queryKey: ["cases", "overview"],
    queryFn: () => api.cases.list({ limit: 100, offset: 0 }),
    refetchInterval: 30_000,
  });
  const monitoringQuery = useQuery({
    queryKey: ["monitoring", "status"],
    queryFn: api.monitoring.status,
    refetchInterval: 60_000,
  });
  const modelQuery = useQuery({
    queryKey: ["models", "active"],
    queryFn: api.models.active,
    staleTime: 60_000,
  });

  const cases = casesQuery.data ?? [];
  const unassigned = cases.filter(
    (caseItem) => caseItem.status === "open" && !caseItem.assigned_to_id,
  ).length;
  const underway = cases.filter((caseItem) =>
    ["claimed", "in_review"].includes(caseItem.status),
  ).length;
  const resolved = cases.filter((caseItem) =>
    ["escalated", "dismissed", "closed"].includes(caseItem.status),
  ).length;
  const recentCases = cases.slice(0, 6);
  const allPending =
    casesQuery.isPending && monitoringQuery.isPending && modelQuery.isPending;
  const allFailed = casesQuery.isError && monitoringQuery.isError && modelQuery.isError;

  const retryAll = () => {
    void casesQuery.refetch();
    void monitoringQuery.refetch();
    void modelQuery.refetch();
  };

  if (allPending) {
    return (
      <div className="page overview-page">
        <PageState
          kind="loading"
          title="Preparing the command center"
          message="Loading the current queue, drift signal, and active model."
        />
      </div>
    );
  }

  if (allFailed) {
    return (
      <div className="page overview-page">
        <PageState
          kind="error"
          title="The command center is unavailable"
          message={errorMessage(casesQuery.error ?? monitoringQuery.error ?? modelQuery.error)}
          onRetry={retryAll}
        />
      </div>
    );
  }

  return (
    <div className="page overview-page">
      <header className="page-header">
        <div>
          <p className="page-eyebrow">Operations overview</p>
          <h2 className="page-title">Good decisions start with a clear queue.</h2>
          <p className="page-subtitle">
            A current operational view of alert ownership, model health, and human review.
          </p>
        </div>
        <div className="page-header-actions">
          <span className="current-view-note">
            <Clock3 size={16} aria-hidden="true" /> Current API view · up to 100 cases
          </span>
          <Link className="button button-primary" to="/cases">
            Open queue <ArrowRight size={16} aria-hidden="true" />
          </Link>
        </div>
      </header>

      {casesQuery.isError || monitoringQuery.isError || modelQuery.isError ? (
        <div className="inline-alert inline-alert-warning" role="status">
          <CircleAlert size={18} aria-hidden="true" />
          <span>
            Some panels could not refresh. Available data remains visible; retry the affected panel below.
          </span>
        </div>
      ) : null}

      <section className="metric-grid" aria-label="Current queue metrics">
        <article className="metric-card">
          <span className="metric-icon metric-icon-blue" aria-hidden="true">
            <BriefcaseBusiness size={20} />
          </span>
          <div>
            <p>Cases in view</p>
            <strong>{casesQuery.data ? formatCompactNumber(cases.length) : "—"}</strong>
            <small>Most recent ranked page</small>
          </div>
        </article>
        <article className="metric-card">
          <span className="metric-icon metric-icon-amber" aria-hidden="true">
            <ScanSearch size={20} />
          </span>
          <div>
            <p>Awaiting owner</p>
            <strong>{casesQuery.data ? formatCompactNumber(unassigned) : "—"}</strong>
            <small>Open and unassigned in view</small>
          </div>
        </article>
        <article className="metric-card">
          <span className="metric-icon metric-icon-violet" aria-hidden="true">
            <Gauge size={20} />
          </span>
          <div>
            <p>Reviews underway</p>
            <strong>{casesQuery.data ? formatCompactNumber(underway) : "—"}</strong>
            <small>Claimed or in review</small>
          </div>
        </article>
        <article className="metric-card">
          <span className="metric-icon metric-icon-green" aria-hidden="true">
            <ShieldCheck size={20} />
          </span>
          <div>
            <p>Resolved in view</p>
            <strong>{casesQuery.data ? formatCompactNumber(resolved) : "—"}</strong>
            <small>Escalated, dismissed, or closed</small>
          </div>
        </article>
      </section>

      <section className="overview-grid">
        <article className="panel overview-queue-panel">
          <div className="panel-header">
            <div>
              <p className="panel-eyebrow">Priority work</p>
              <h3>Recent queue</h3>
              <p>Ordered by the backend queue rank for the current page.</p>
            </div>
            <Link className="text-link" to="/cases">
              View all <ArrowRight size={15} aria-hidden="true" />
            </Link>
          </div>

          {casesQuery.isPending ? (
            <PageState kind="loading" title="Loading the queue" />
          ) : casesQuery.isError ? (
            <PageState
              kind="error"
              title="Queue unavailable"
              message={errorMessage(casesQuery.error)}
              onRetry={() => void casesQuery.refetch()}
            />
          ) : recentCases.length === 0 ? (
            <PageState
              kind="empty"
              title="No cases in the current view"
              message="The queue will appear here after alerts are materialized."
            />
          ) : (
            <div className="data-table-wrap">
              <table className="data-table">
                <caption className="sr-only">Six most recent cases from the current queue page</caption>
                <thead>
                  <tr>
                    <th scope="col">Rank</th>
                    <th scope="col">Case</th>
                    <th scope="col">Risk</th>
                    <th scope="col">Status</th>
                    <th scope="col">Updated</th>
                    <th scope="col"><span className="sr-only">Open case</span></th>
                  </tr>
                </thead>
                <tbody>
                  {recentCases.map((caseItem) => {
                    const presentation = getStudyArmPresentation(caseItem.study_arm);
                    return (
                      <tr key={caseItem.id}>
                        <td><span className="queue-rank">#{caseItem.queue_rank}</span></td>
                        <td>
                          <strong>{caseItem.review_id}</strong>
                          <small>{formatDateTime(caseItem.alert_hour)}</small>
                        </td>
                        <td>
                          {presentation.showScore ? (
                            <span className={`risk-value risk-value-${riskTone(caseItem.risk_score)}`}>
                              {formatPercent(caseItem.risk_score, 1)}
                            </span>
                          ) : (
                            <span
                              className="risk-value-withheld"
                              aria-label="Risk score withheld by assigned evidence view"
                            >
                              Withheld
                            </span>
                          )}
                        </td>
                        <td>
                          <span className={`status-badge status-badge-${caseTone(caseItem)}`}>
                            {caseStatusLabel(caseItem.status)}
                          </span>
                        </td>
                        <td title={formatDateTime(caseItem.updated_at)}>
                          {formatRelativeTime(caseItem.updated_at)}
                        </td>
                        <td>
                          <Link
                            className="icon-link"
                            to={`/cases/${caseItem.id}`}
                            aria-label={`Open case ${caseItem.review_id}`}
                          >
                            <ArrowRight size={17} aria-hidden="true" />
                          </Link>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </article>

        <div className="overview-side-stack">
          <article className="panel health-panel">
            <div className="panel-header">
              <div>
                <p className="panel-eyebrow">Pattern monitoring</p>
                <h3>Drift status</h3>
              </div>
              <Activity size={20} aria-hidden="true" />
            </div>
            {monitoringQuery.isPending ? (
              <PageState kind="loading" title="Loading drift status" />
            ) : monitoringQuery.isError ? (
              <PageState
                kind="error"
                title="Drift status unavailable"
                message={errorMessage(monitoringQuery.error)}
                onRetry={() => void monitoringQuery.refetch()}
              />
            ) : (
              <div className="health-summary">
                <span className={`status-badge status-badge-${driftTone(monitoringQuery.data.status)}`}>
                  {monitoringQuery.data.status ? humanize(monitoringQuery.data.status) : "No snapshot"}
                </span>
                <strong>{monitoringQuery.data.active_model_version}</strong>
                <p>
                  {monitoringQuery.data.latest_snapshot
                    ? `Window ended ${formatDateTime(monitoringQuery.data.latest_snapshot.window_end)}.`
                    : "The active model has no persisted drift snapshot yet."}
                </p>
                <Link className="text-link" to="/monitoring">
                  Inspect monitoring <ArrowRight size={15} aria-hidden="true" />
                </Link>
              </div>
            )}
          </article>

          <article className="panel health-panel">
            <div className="panel-header">
              <div>
                <p className="panel-eyebrow">Governed asset</p>
                <h3>Active model</h3>
              </div>
              <Database size={20} aria-hidden="true" />
            </div>
            {modelQuery.isPending ? (
              <PageState kind="loading" title="Loading model status" />
            ) : modelQuery.isError ? (
              <PageState
                kind="error"
                title="Model status unavailable"
                message={errorMessage(modelQuery.error)}
                onRetry={() => void modelQuery.refetch()}
              />
            ) : (
              <div className="health-summary">
                <span className="status-badge status-badge-success">{humanize(modelQuery.data.status)}</span>
                <strong>{modelQuery.data.name} · {modelQuery.data.version}</strong>
                <p>
                  {humanize(modelQuery.data.algorithm)}
                  {modelQuery.data.activated_at
                    ? ` · activated ${formatRelativeTime(modelQuery.data.activated_at)}`
                    : " · activation time unavailable"}
                </p>
                {user?.role === "admin" ? (
                  <Link className="text-link" to="/governance">
                    View governance <ArrowRight size={15} aria-hidden="true" />
                  </Link>
                ) : (
                  <span className="assurance-line">
                    <ShieldCheck size={14} aria-hidden="true" /> Registry-controlled active version
                  </span>
                )}
              </div>
            )}
          </article>
        </div>
      </section>

      <section className="panel workflow-panel" aria-labelledby="workflow-title">
        <div className="panel-header">
          <div>
            <p className="panel-eyebrow">Investigation workflow</p>
            <h3 id="workflow-title">From transaction signal to human decision</h3>
            <p>The model prioritizes work; authorized analysts retain the final decision.</p>
          </div>
        </div>
        <ol className="workflow-steps">
          <li>
            <span>01</span>
            <Database size={20} aria-hidden="true" />
            <div><strong>Transaction intake</strong><p>Masked transaction records enter the governed API.</p></div>
          </li>
          <li>
            <span>02</span>
            <Gauge size={20} aria-hidden="true" />
            <div><strong>Risk prioritization</strong><p>Signals become a finite, ranked analyst queue.</p></div>
          </li>
          <li>
            <span>03</span>
            <ScanSearch size={20} aria-hidden="true" />
            <div><strong>Human review</strong><p>An analyst examines context and records a rationale.</p></div>
          </li>
          <li>
            <span>04</span>
            <Activity size={20} aria-hidden="true" />
            <div><strong>Ongoing oversight</strong><p>KS and ADWIN signals surface pattern change for review.</p></div>
          </li>
        </ol>
      </section>
    </div>
  );
}
