import { FormEvent, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  ClipboardCheck,
  Fingerprint,
  LockKeyhole,
  RefreshCw,
  Search,
  ShieldCheck,
  X,
} from "lucide-react";
import { useAuth } from "../../auth/AuthContext";
import { PageState } from "../../components/ui/PageState";
import { api, normalizeApiError } from "../../lib/api";
import { formatDateTime, formatPercent, humanize } from "../../lib/format";
import type { JsonValue, ModelVersion } from "../../lib/types";
import "../dashboard/operations-shared.css";
import "./governance.css";

type AuditFilters = {
  eventType: string;
  resourceType: string;
};

type MetricRow = {
  key: string;
  value: string | number | boolean;
};

function shortId(value: string | null | undefined, length = 12) {
  if (!value) return "—";
  return value.length > length ? `${value.slice(0, length)}…` : value;
}

function featureCount(model: ModelVersion): number | null {
  const names = model.feature_schema.feature_names;
  if (Array.isArray(names)) return names.length;
  const features = model.feature_schema.features;
  if (Array.isArray(features)) return features.length;
  const count = model.feature_schema.feature_count;
  return typeof count === "number" && Number.isFinite(count) ? count : null;
}

function scalarMetricRows(
  value: Record<string, JsonValue>,
  prefix = "",
  rows: MetricRow[] = [],
): MetricRow[] {
  for (const [key, item] of Object.entries(value)) {
    if (rows.length >= 10) break;
    const path = prefix ? `${prefix} · ${key}` : key;
    if (item !== null && typeof item === "object" && !Array.isArray(item)) {
      scalarMetricRows(item, path, rows);
    } else if (["string", "number", "boolean"].includes(typeof item)) {
      rows.push({ key: path, value: item as string | number | boolean });
    }
  }
  return rows;
}

function displayMetric(key: string, value: string | number | boolean) {
  if (typeof value !== "number") return humanize(String(value));
  if (/auc|precision|recall|f1|rate|accuracy|prevalence/i.test(key) && value >= 0 && value <= 1) {
    return formatPercent(value, 2);
  }
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: 4 }).format(value);
}

function detailEntries(details: Record<string, JsonValue>) {
  return Object.entries(details).filter(([, value]) =>
    value === null || ["string", "number", "boolean"].includes(typeof value),
  );
}

export function GovernancePage() {
  const { user } = useAuth();
  const isAdmin = user?.role === "admin";
  const [draftFilters, setDraftFilters] = useState<AuditFilters>({
    eventType: "",
    resourceType: "",
  });
  const [appliedFilters, setAppliedFilters] = useState<AuditFilters>({
    eventType: "",
    resourceType: "",
  });
  const [page, setPage] = useState(0);
  const pageSize = 25;

  const modelQuery = useQuery({
    queryKey: ["models", "active"],
    queryFn: api.models.active,
    staleTime: 60_000,
  });
  const auditQuery = useQuery({
    queryKey: ["audit", appliedFilters, page, pageSize],
    queryFn: () =>
      api.audit.list({
        event_type: appliedFilters.eventType.trim() || undefined,
        resource_type: appliedFilters.resourceType.trim() || undefined,
        limit: pageSize,
        offset: page * pageSize,
      }),
    enabled: isAdmin,
    staleTime: 15_000,
  });

  const applyFilters = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setAppliedFilters({
      eventType: draftFilters.eventType.trim(),
      resourceType: draftFilters.resourceType.trim(),
    });
    setPage(0);
  };
  const clearFilters = () => {
    const empty = { eventType: "", resourceType: "" };
    setDraftFilters(empty);
    setAppliedFilters(empty);
    setPage(0);
  };
  const refresh = () => {
    void modelQuery.refetch();
    if (isAdmin) void auditQuery.refetch();
  };

  const model = modelQuery.data;
  const metrics = model ? scalarMetricRows(model.metrics) : [];
  const auditEvents = auditQuery.data ?? [];
  const hasAuditFilters = Boolean(appliedFilters.eventType || appliedFilters.resourceType);
  const auditStart = auditEvents.length > 0 ? page * pageSize + 1 : 0;
  const auditEnd = page * pageSize + auditEvents.length;
  const refreshing = modelQuery.isFetching || (isAdmin && auditQuery.isFetching);

  return (
    <div className="page governance-page">
      <header className="page-header">
        <div>
          <p className="page-eyebrow">Controlled evidence</p>
          <h2 className="page-title">Governance &amp; audit</h2>
          <p className="page-subtitle">
            Read-only evidence for the active model and sanitized, append-only operational events.
          </p>
        </div>
        <div className="page-header-actions">
          <span className="status-badge status-badge-neutral">Read-only view</span>
          <button
            className="button button-secondary"
            type="button"
            onClick={refresh}
            disabled={refreshing}
          >
            <RefreshCw className={refreshing ? "spin" : ""} size={16} aria-hidden="true" />
            {refreshing ? "Refreshing…" : "Refresh"}
          </button>
        </div>
      </header>

      <section className="governance-model-grid" aria-label="Active model governance summary">
        <article className="panel model-summary-panel">
          <div className="panel-header">
            <div>
              <p className="panel-eyebrow">Registry evidence</p>
              <h3>Active model</h3>
              <p>The model currently used by inference, as reported by the backend registry.</p>
            </div>
            <ShieldCheck size={22} aria-hidden="true" />
          </div>

          {modelQuery.isPending ? (
            <PageState kind="loading" title="Loading active model" />
          ) : modelQuery.isError ? (
            <PageState
              kind="error"
              title="Active model unavailable"
              message={normalizeApiError(modelQuery.error)}
              onRetry={() => void modelQuery.refetch()}
            />
          ) : model ? (
            <>
              <div className="model-identity">
                <span className="model-mark" aria-hidden="true"><Fingerprint size={24} /></span>
                <div>
                  <span className="status-badge status-badge-success">{humanize(model.status)}</span>
                  <h4>{model.name}</h4>
                  <p>{model.version} · {humanize(model.algorithm)}</p>
                </div>
              </div>
              <dl className="model-facts">
                <div>
                  <dt>Activated</dt>
                  <dd>{formatDateTime(model.activated_at)}</dd>
                </div>
                <div>
                  <dt>Registered</dt>
                  <dd>{formatDateTime(model.created_at)}</dd>
                </div>
                <div>
                  <dt>Feature schema</dt>
                  <dd>{featureCount(model) ?? "Count unavailable"}</dd>
                </div>
                <div>
                  <dt>Artifact digest</dt>
                  <dd title={model.artifact_sha256}><code>{shortId(model.artifact_sha256, 18)}</code></dd>
                </div>
              </dl>
            </>
          ) : null}
        </article>

        <article className="panel governance-controls-panel">
          <div className="panel-header">
            <div>
              <p className="panel-eyebrow">Control posture</p>
              <h3>Lifecycle safeguards</h3>
            </div>
            <ClipboardCheck size={21} aria-hidden="true" />
          </div>
          <ul className="control-list">
            <li>
              <CheckCircle2 size={18} aria-hidden="true" />
              <span><strong>Human-gated changes</strong><small>No automatic model promotion is exposed here.</small></span>
            </li>
            <li>
              <CheckCircle2 size={18} aria-hidden="true" />
              <span><strong>Artifact integrity</strong><small>The registry records a SHA-256 digest for the active bundle.</small></span>
            </li>
            <li>
              <CheckCircle2 size={18} aria-hidden="true" />
              <span><strong>Sanitized evidence</strong><small>Only backend-approved safe audit details are rendered.</small></span>
            </li>
          </ul>
        </article>
      </section>

      <section className="panel model-metrics-panel" aria-labelledby="model-metrics-title">
        <div className="panel-header">
          <div>
            <p className="panel-eyebrow">Registered evidence</p>
            <h3 id="model-metrics-title">Model metrics</h3>
            <p>Scalar metrics present in the active model record; values are not recomputed by the browser.</p>
          </div>
        </div>
        {modelQuery.isPending ? (
          <PageState kind="loading" title="Loading model metrics" />
        ) : modelQuery.isError ? (
          <p className="muted-copy">Metrics are unavailable while the active model request is failing.</p>
        ) : metrics.length === 0 ? (
          <PageState
            kind="empty"
            title="No scalar metrics registered"
            message="The model is active, but this record does not contain displayable scalar metrics."
          />
        ) : (
          <dl className="metric-definition-grid">
            {metrics.map((metric) => (
              <div key={metric.key}>
                <dt>{humanize(metric.key)}</dt>
                <dd>{displayMetric(metric.key, metric.value)}</dd>
              </div>
            ))}
          </dl>
        )}
      </section>

      <section className="panel audit-panel" aria-labelledby="audit-title">
        <div className="panel-header audit-header">
          <div>
            <p className="panel-eyebrow">Administrator access</p>
            <h3 id="audit-title">Operational audit trail</h3>
            <p>Exact-match server filters over sanitized events, newest first.</p>
          </div>
          <LockKeyhole size={21} aria-hidden="true" />
        </div>

        {!isAdmin ? (
          <div className="permission-state">
            <span className="permission-state-icon" aria-hidden="true"><LockKeyhole size={20} /></span>
            <div>
              <strong>Administrator access required</strong>
              <p>Your current role is not permitted to retrieve the audit trail.</p>
            </div>
          </div>
        ) : (
          <>
            <form className="filters audit-filters" onSubmit={applyFilters}>
              <label className="search-field" htmlFor="audit-event-filter">
                <span>Event type</span>
                <span className="search-input-wrap">
                  <Search size={16} aria-hidden="true" />
                  <input
                    id="audit-event-filter"
                    value={draftFilters.eventType}
                    placeholder="case.claimed"
                    onChange={(event) =>
                      setDraftFilters((current) => ({ ...current, eventType: event.target.value }))
                    }
                  />
                </span>
              </label>
              <label className="search-field" htmlFor="audit-resource-filter">
                <span>Resource type</span>
                <span className="search-input-wrap">
                  <Search size={16} aria-hidden="true" />
                  <input
                    id="audit-resource-filter"
                    value={draftFilters.resourceType}
                    placeholder="case"
                    onChange={(event) =>
                      setDraftFilters((current) => ({ ...current, resourceType: event.target.value }))
                    }
                  />
                </span>
              </label>
              <button className="button button-primary" type="submit">Apply filters</button>
              {hasAuditFilters ? (
                <button className="button button-ghost" type="button" onClick={clearFilters}>
                  <X size={16} aria-hidden="true" /> Clear
                </button>
              ) : null}
            </form>

            {auditQuery.isPending ? (
              <PageState kind="loading" title="Loading audit events" />
            ) : auditQuery.isError ? (
              <PageState
                kind="error"
                title="Audit trail unavailable"
                message={normalizeApiError(auditQuery.error)}
                onRetry={() => void auditQuery.refetch()}
              />
            ) : auditEvents.length === 0 ? (
              <PageState
                kind="empty"
                title={hasAuditFilters ? "No events match these filters" : "No audit events available"}
                message={
                  hasAuditFilters
                    ? "Clear or change the exact event and resource values."
                    : "Sanitized operational events will appear here when activity is recorded."
                }
              />
            ) : (
              <div className="data-table-wrap">
                <table className="data-table audit-data-table">
                  <caption className="sr-only">
                    Sanitized audit events {auditStart} through {auditEnd}
                  </caption>
                  <thead>
                    <tr>
                      <th scope="col">Time</th>
                      <th scope="col">Event</th>
                      <th scope="col">Actor</th>
                      <th scope="col">Resource</th>
                      <th scope="col">Request</th>
                      <th scope="col">Safe details</th>
                    </tr>
                  </thead>
                  <tbody>
                    {auditEvents.map((event) => {
                      const details = detailEntries(event.safe_details);
                      return (
                        <tr key={event.id}>
                          <td>{formatDateTime(event.created_at)}</td>
                          <td><strong>{event.event_type}</strong></td>
                          <td title={event.actor_user_id ?? undefined}>
                            {event.actor_user_id ? <code>{shortId(event.actor_user_id)}</code> : "System"}
                          </td>
                          <td>
                            <span>{humanize(event.resource_type)}</span>
                            {event.resource_id ? (
                              <small title={event.resource_id}>{shortId(event.resource_id)}</small>
                            ) : null}
                          </td>
                          <td title={event.request_id ?? undefined}>
                            {event.request_id ? <code>{shortId(event.request_id)}</code> : "—"}
                          </td>
                          <td>
                            {details.length === 0 ? (
                              "—"
                            ) : (
                              <details className="audit-details">
                                <summary>{details.length} field{details.length === 1 ? "" : "s"}</summary>
                                <dl>
                                  {details.map(([key, value]) => (
                                    <div key={key}>
                                      <dt>{humanize(key)}</dt>
                                      <dd>{String(value ?? "—")}</dd>
                                    </div>
                                  ))}
                                </dl>
                              </details>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}

            {!auditQuery.isError && auditEvents.length > 0 ? (
              <nav className="pagination" aria-label="Audit event pages">
                <p>Showing API rows <strong>{auditStart}–{auditEnd}</strong></p>
                <div>
                  <button
                    className="button button-secondary button-compact"
                    type="button"
                    disabled={page === 0 || auditQuery.isFetching}
                    onClick={() => setPage((current) => Math.max(0, current - 1))}
                  >
                    <ChevronLeft size={16} aria-hidden="true" /> Previous
                  </button>
                  <span aria-current="page">Page {page + 1}</span>
                  <button
                    className="button button-secondary button-compact"
                    type="button"
                    disabled={auditEvents.length < pageSize || auditQuery.isFetching}
                    onClick={() => setPage((current) => current + 1)}
                  >
                    Next <ChevronRight size={16} aria-hidden="true" />
                  </button>
                </div>
              </nav>
            ) : null}
          </>
        )}
      </section>
    </div>
  );
}
