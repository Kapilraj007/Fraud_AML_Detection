import { useQuery } from "@tanstack/react-query";
import { RefreshCw } from "lucide-react";
import { useAuth } from "../../auth/AuthContext";
import { api, normalizeApiError } from "../../lib/api";
import { formatDateTime } from "../../lib/format";
import { DriftDashboard } from "./DriftDashboard";
import { RetrainTriggerPanel } from "./RetrainTriggerPanel";
import "../dashboard/operations-shared.css";
import "./monitoring.css";

export function MonitoringPage() {
  const { user } = useAuth();
  const statusQuery = useQuery({
    queryKey: ["monitoring", "status"],
    queryFn: api.monitoring.status,
    refetchInterval: 60_000,
  });
  const activeModelId = statusQuery.data?.active_model_id;
  const historyQuery = useQuery({
    queryKey: ["monitoring", "history", activeModelId, 100],
    queryFn: () =>
      activeModelId
        ? api.monitoring.history({ model_version_id: activeModelId, limit: 100, offset: 0 })
        : Promise.resolve([]),
    enabled: Boolean(activeModelId),
    refetchInterval: 60_000,
  });

  // Keep the display safe even if a proxy or older API ignores the model filter.
  const activeModelHistory = activeModelId
    ? (historyQuery.data ?? []).filter(
        (snapshot) => snapshot.model_version_id === activeModelId,
      )
    : [];
  const reportedStatusSnapshot = statusQuery.data?.latest_snapshot;
  const statusSnapshot =
    activeModelId && reportedStatusSnapshot?.model_version_id === activeModelId
      ? reportedStatusSnapshot
      : null;
  const latestSnapshot = statusSnapshot ?? activeModelHistory[0] ?? null;
  const statusError = statusQuery.isError
    ? normalizeApiError(statusQuery.error, "The current drift posture could not be loaded.")
    : null;
  const historyError = statusError ?? (historyQuery.isError
    ? normalizeApiError(historyQuery.error, "Monitoring history could not be loaded.")
    : null);
  const historyLoading =
    statusQuery.isPending || (Boolean(activeModelId) && historyQuery.isPending);

  const refresh = () => {
    void statusQuery.refetch();
    if (activeModelId) void historyQuery.refetch();
  };
  const refreshing =
    statusQuery.isFetching || (Boolean(activeModelId) && historyQuery.isFetching);
  const lastObserved = latestSnapshot?.window_end;

  return (
    <div className="page monitoring-page">
      <header className="page-header">
        <div>
          <p className="page-eyebrow">Model oversight</p>
          <h2 className="page-title">Risk monitoring</h2>
          <p className="page-subtitle">
            Review feature-distribution change and the label-independent score stream before taking action.
          </p>
        </div>
        <div className="page-header-actions">
          {lastObserved ? (
            <span className="current-view-note">Last observed {formatDateTime(lastObserved)}</span>
          ) : null}
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

      <DriftDashboard
        status={statusQuery.data}
        history={activeModelHistory}
        statusLoading={statusQuery.isPending}
        historyLoading={historyLoading}
        statusError={statusError}
        historyError={historyError}
        onRetryStatus={() => void statusQuery.refetch()}
        onRetryHistory={() => {
          if (statusError) {
            void statusQuery.refetch();
          } else {
            void historyQuery.refetch();
          }
        }}
      />

      <RetrainTriggerPanel
        snapshot={latestSnapshot}
        snapshotLoading={!latestSnapshot && historyLoading}
        snapshotError={!latestSnapshot ? historyError : null}
        onRetrySnapshot={refresh}
        role={user?.role}
      />
    </div>
  );
}
