import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Activity, CheckCircle2, RadioTower, TriangleAlert } from "lucide-react";
import { PageState } from "../../components/ui/PageState";
import { formatDateTime, humanize } from "../../lib/format";
import type { DriftSnapshot, DriftStatus, MonitoringStatus } from "../../lib/types";
import { extractKsReadings, monitoredFeatures } from "./driftMetrics";

type DriftDashboardProps = {
  status: MonitoringStatus | undefined;
  history: DriftSnapshot[];
  statusLoading: boolean;
  historyLoading: boolean;
  statusError: string | null;
  historyError: string | null;
  onRetryStatus: () => void;
  onRetryHistory: () => void;
};

const chartColors = ["#2457d6", "#0f766e", "#9a3412", "#7e22ce", "#be123c", "#3f6212"];

function statusTone(status: DriftStatus | null | undefined) {
  if (status === "drift_detected") return "critical";
  if (status === "watch") return "warning";
  if (status === "stable") return "success";
  return "neutral";
}

function formatKs(value: number | null) {
  return value === null ? "—" : value.toFixed(3);
}

function formatPValue(value: number | null) {
  if (value === null) return null;
  return value < 0.001 ? "p < 0.001" : `p = ${value.toFixed(3)}`;
}

function metricSummary(snapshot: DriftSnapshot | null) {
  if (!snapshot) return [];
  return Object.entries(snapshot.prediction_metrics)
    .filter(([, value]) => ["string", "number", "boolean"].includes(typeof value))
    .slice(0, 4);
}

export function DriftDashboard({
  status,
  history,
  statusLoading,
  historyLoading,
  statusError,
  historyError,
  onRetryStatus,
  onRetryHistory,
}: DriftDashboardProps) {
  const reportedStatusSnapshot = status?.latest_snapshot;
  const latest =
    status && reportedStatusSnapshot?.model_version_id === status.active_model_id
      ? reportedStatusSnapshot
      : history[0] ?? null;
  const chartHistory = [...history]
    .sort((left, right) => Date.parse(left.window_end) - Date.parse(right.window_end))
    .slice(-30);
  const features = monitoredFeatures(chartHistory);
  const chartData = chartHistory.map((snapshot) => {
    const point: Record<string, string | number | null> = {
      window: snapshot.window_end,
      label: new Intl.DateTimeFormat(undefined, {
        month: "short",
        day: "numeric",
        hour: "numeric",
      }).format(new Date(snapshot.window_end)),
    };
    for (const reading of extractKsReadings(snapshot)) {
      if (features.includes(reading.feature)) point[reading.feature] = reading.statistic;
    }
    return point;
  });
  const tableHistory = [...history]
    .sort((left, right) => Date.parse(right.window_end) - Date.parse(left.window_end))
    .slice(0, 12);
  const latestPredictionMetrics = metricSummary(latest);

  return (
    <>
      <section className="monitoring-summary-grid" aria-label="Current monitoring state">
        <article className="panel monitoring-status-panel">
          <div className="panel-header">
            <div>
              <p className="panel-eyebrow">Current signal</p>
              <h3>Drift posture</h3>
            </div>
            <Activity size={20} aria-hidden="true" />
          </div>
          {statusLoading ? (
            <PageState kind="loading" title="Loading drift posture" />
          ) : statusError ? (
            <PageState
              kind="error"
              title="Drift posture unavailable"
              message={statusError}
              onRetry={onRetryStatus}
            />
          ) : status ? (
            <div className="monitoring-status-body">
              <span className={`status-badge status-badge-${statusTone(status.status)}`}>
                {status.status ? humanize(status.status) : "Awaiting snapshot"}
              </span>
              <strong>{status.active_model_version}</strong>
              <p>
                {latest
                  ? `Latest observed window ended ${formatDateTime(latest.window_end)}.`
                  : "The active model has not produced a drift snapshot yet."}
              </p>
            </div>
          ) : null}
        </article>

        <article className="panel monitoring-status-panel">
          <div className="panel-header">
            <div>
              <p className="panel-eyebrow">Score stream</p>
              <h3>ADWIN detector</h3>
            </div>
            <RadioTower size={20} aria-hidden="true" />
          </div>
          {statusLoading && !latest ? (
            <PageState kind="loading" title="Loading ADWIN state" />
          ) : latest ? (
            <div className="adwin-state">
              <span
                className={`adwin-indicator ${latest.adwin_detected ? "is-alert" : "is-stable"}`}
                aria-hidden="true"
              >
                {latest.adwin_detected ? <TriangleAlert size={22} /> : <CheckCircle2 size={22} />}
              </span>
              <div>
                <strong>{latest.adwin_detected ? "Change detected" : "No change detected"}</strong>
                <p>
                  Latest label-independent score-stream reading. It is a review signal, not an
                  automatic retraining or deployment decision.
                </p>
              </div>
            </div>
          ) : (
            <PageState
              kind="empty"
              title="No ADWIN reading yet"
              message="A reading appears after the first monitoring window is persisted."
            />
          )}
        </article>

        <article className="panel monitoring-status-panel prediction-metrics-panel">
          <div className="panel-header">
            <div>
              <p className="panel-eyebrow">Latest window</p>
              <h3>Prediction metrics</h3>
            </div>
          </div>
          {latestPredictionMetrics.length > 0 ? (
            <dl className="compact-definition-list">
              {latestPredictionMetrics.map(([key, value]) => (
                <div key={key}>
                  <dt>{humanize(key)}</dt>
                  <dd>{String(value)}</dd>
                </div>
              ))}
            </dl>
          ) : (
            <p className="muted-copy">No scalar prediction metrics were stored for this window.</p>
          )}
        </article>
      </section>

      <section className="panel drift-trend-panel" aria-labelledby="ks-trend-title">
        <div className="panel-header">
          <div>
            <p className="panel-eyebrow">Feature distribution</p>
            <h3 id="ks-trend-title">KS statistic by monitoring window</h3>
            <p>
              Up to six persisted feature series across the 30 most recent windows. Higher values
              indicate a larger distribution difference; alert thresholds are managed by the service.
            </p>
          </div>
        </div>

        {historyLoading ? (
          <PageState kind="loading" title="Loading feature trends" />
        ) : historyError ? (
          <PageState
            kind="error"
            title="Feature trends unavailable"
            message={historyError}
            onRetry={onRetryHistory}
          />
        ) : history.length === 0 ? (
          <PageState
            kind="empty"
            title="No monitoring history"
            message="KS trend lines will appear after snapshots are recorded."
          />
        ) : features.length === 0 ? (
          <PageState
            kind="empty"
            title="No KS statistics in these snapshots"
            message="Monitoring windows exist, but their feature payloads do not contain chartable KS values."
          />
        ) : (
          <>
            <figure className="chart-figure">
              <div className="chart-container" role="img" aria-label="Feature KS statistics over time">
                <ResponsiveContainer width="100%" height={340}>
                  <LineChart
                    data={chartData}
                    margin={{ top: 12, right: 18, left: 0, bottom: 8 }}
                    accessibilityLayer
                  >
                    <CartesianGrid strokeDasharray="4 4" vertical={false} />
                    <XAxis
                      dataKey="window"
                      minTickGap={28}
                      tickFormatter={(value: string) =>
                        new Intl.DateTimeFormat(undefined, {
                          month: "short",
                          day: "numeric",
                          hour: "numeric",
                        }).format(new Date(value))
                      }
                    />
                    <YAxis domain={[0, 1]} tickFormatter={(value: number) => value.toFixed(1)} />
                    <Tooltip labelFormatter={(value: unknown) => formatDateTime(String(value))} />
                    <Legend formatter={(value: string) => humanize(value)} />
                    {features.map((feature, index) => (
                      <Line
                        key={feature}
                        dataKey={feature}
                        name={feature}
                        stroke={chartColors[index % chartColors.length]}
                        strokeWidth={2.25}
                        dot={{ r: 3 }}
                        activeDot={{ r: 5 }}
                        connectNulls={false}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <figcaption>
                KS values range from 0 to 1. Exact readings and p-values are available in the table below.
              </figcaption>
            </figure>

            <details className="data-alternative" open>
              <summary>Exact KS readings · 12 most recent windows</summary>
              <div className="data-table-wrap">
                <table className="data-table monitoring-data-table">
                  <caption className="sr-only">
                    Exact Kolmogorov-Smirnov statistics corresponding to the trend chart
                  </caption>
                  <thead>
                    <tr>
                      <th scope="col">Window end</th>
                      <th scope="col">Status</th>
                      <th scope="col">ADWIN</th>
                      {features.map((feature) => (
                        <th key={feature} scope="col">{humanize(feature)}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {tableHistory.map((snapshot) => {
                      const readings = new Map(
                        extractKsReadings(snapshot).map((reading) => [reading.feature, reading]),
                      );
                      return (
                        <tr key={snapshot.id}>
                          <td>{formatDateTime(snapshot.window_end)}</td>
                          <td>
                            <span className={`status-badge status-badge-${statusTone(snapshot.status)}`}>
                              {humanize(snapshot.status)}
                            </span>
                          </td>
                          <td>{snapshot.adwin_detected ? "Change" : "No change"}</td>
                          {features.map((feature) => {
                            const reading = readings.get(feature);
                            const pValue = reading ? formatPValue(reading.pValue) : null;
                            return (
                              <td key={feature}>
                                {formatKs(reading?.statistic ?? null)}
                                {pValue ? <small>{pValue}</small> : null}
                              </td>
                            );
                          })}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </details>
          </>
        )}
      </section>
    </>
  );
}
