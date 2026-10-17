import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { DriftSnapshot, MonitoringStatus } from "../../lib/types";
import { MonitoringPage } from "./MonitoringPage";

const monitoringMocks = vi.hoisted(() => ({
  status: vi.fn(),
  history: vi.fn(),
}));

vi.mock("../../auth/AuthContext", () => ({
  useAuth: () => ({ user: { role: "admin" } }),
}));

vi.mock("../../lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../lib/api")>();
  return {
    ...actual,
    api: {
      ...actual.api,
      monitoring: {
        ...actual.api.monitoring,
        status: monitoringMocks.status,
        history: monitoringMocks.history,
      },
    },
  };
});

vi.mock("./DriftDashboard", () => ({
  DriftDashboard: ({
    history,
    historyLoading,
  }: {
    history: DriftSnapshot[];
    historyLoading: boolean;
  }) => (
    <div data-testid="drift-dashboard" data-loading={String(historyLoading)}>
      {history.map((snapshot) => snapshot.id).join(",")}
    </div>
  ),
}));

function driftSnapshot(id: string, modelVersionId: string): DriftSnapshot {
  return {
    id,
    model_version_id: modelVersionId,
    window_start: "2026-10-14T08:00:00Z",
    window_end: "2026-10-14T09:00:00Z",
    status: "stable",
    adwin_detected: false,
    feature_metrics: {},
    prediction_metrics: {},
    created_at: "2026-10-14T09:01:00Z",
  };
}

function renderPage() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <MonitoringPage />
    </QueryClientProvider>,
  );
}

afterEach(() => {
  monitoringMocks.status.mockReset();
  monitoringMocks.history.mockReset();
});

describe("MonitoringPage active-model integrity", () => {
  it("requests and displays history only for the active model", async () => {
    const activeSnapshot = driftSnapshot("active-snapshot", "active-model");
    const retiredSnapshot = driftSnapshot("retired-snapshot", "retired-model");
    const status: MonitoringStatus = {
      active_model_id: "active-model",
      active_model_version: "2026.10",
      status: null,
      latest_snapshot: null,
    };
    monitoringMocks.status.mockResolvedValue(status);
    // Deliberately include a retired record to verify the client-side integrity guard.
    monitoringMocks.history.mockResolvedValue([retiredSnapshot, activeSnapshot]);

    renderPage();

    await waitFor(() => {
      expect(monitoringMocks.history).toHaveBeenCalledWith({
        model_version_id: "active-model",
        limit: 100,
        offset: 0,
      });
    });
    await waitFor(() => {
      expect(screen.getByTestId("drift-dashboard")).toHaveTextContent("active-snapshot");
    });
    expect(screen.getByTestId("drift-dashboard")).not.toHaveTextContent("retired-snapshot");
    expect(await screen.findByText("Snapshot active-snapshot")).toBeInTheDocument();
  });

  it("shows loading and then a precise empty state when the active model has no snapshot", async () => {
    monitoringMocks.status.mockResolvedValue({
      active_model_id: "active-model",
      active_model_version: "2026.10",
      status: null,
      latest_snapshot: null,
    } satisfies MonitoringStatus);
    monitoringMocks.history.mockResolvedValue([]);

    renderPage();

    expect(screen.getByText("Loading active-model snapshot")).toBeInTheDocument();
    expect(await screen.findByText("No active-model snapshot")).toBeInTheDocument();
    expect(screen.getByText(/active model has no persisted monitoring window/i)).toBeInTheDocument();
  });
});
