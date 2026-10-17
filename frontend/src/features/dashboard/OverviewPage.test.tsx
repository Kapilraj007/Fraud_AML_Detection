import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { Case, ModelVersion, MonitoringStatus } from "../../lib/types";

const apiMocks = vi.hoisted(() => ({
  listCases: vi.fn(),
  monitoringStatus: vi.fn(),
  activeModel: vi.fn(),
}));

vi.mock("../../auth/AuthContext", () => ({
  useAuth: () => ({ user: { role: "fraud_analyst" } }),
}));

vi.mock("../../lib/api", () => ({
  api: {
    cases: { list: apiMocks.listCases },
    monitoring: { status: apiMocks.monitoringStatus },
    models: { active: apiMocks.activeModel },
  },
}));

import { OverviewPage } from "./OverviewPage";

const now = "2026-10-14T08:00:00Z";

const cases: Case[] = [
  {
    id: "case-data-only",
    review_id: "DATA-ONLY",
    status: "open",
    queue_rank: 1,
    risk_score: 0.91,
    assigned_to_id: null,
    study_arm: "data_only",
    alert_hour: now,
    updated_at: now,
    created_at: now,
  },
  {
    id: "case-score-only",
    review_id: "SCORE-ONLY",
    status: "open",
    queue_rank: 2,
    risk_score: 0.42,
    assigned_to_id: null,
    study_arm: "score_only",
    alert_hour: now,
    updated_at: now,
    created_at: now,
  },
];

const monitoring: MonitoringStatus = {
  active_model_id: "model-1",
  active_model_version: "v1",
  status: null,
  latest_snapshot: null,
};

const model: ModelVersion = {
  id: "model-1",
  version: "v1",
  name: "Fraud ensemble",
  algorithm: "ensemble",
  artifact_uri: "models/v1",
  artifact_sha256: "abc123",
  feature_schema: {},
  metrics: {},
  status: "active",
  created_at: now,
  activated_at: now,
};

function renderOverview() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        <OverviewPage />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  apiMocks.listCases.mockResolvedValue(cases);
  apiMocks.monitoringStatus.mockResolvedValue(monitoring);
  apiMocks.activeModel.mockResolvedValue(model);
});

describe("command-center study-arm presentation", () => {
  it("withholds case risk for data-only assignments while showing allowed scores", async () => {
    renderOverview();

    const dataOnlyRow = (await screen.findByText("DATA-ONLY")).closest("tr");
    const scoreOnlyRow = screen.getByText("SCORE-ONLY").closest("tr");
    expect(dataOnlyRow).not.toBeNull();
    expect(scoreOnlyRow).not.toBeNull();

    const hiddenRisk = new Intl.NumberFormat(undefined, {
      style: "percent",
      maximumFractionDigits: 1,
    }).format(0.91);
    const visibleRisk = new Intl.NumberFormat(undefined, {
      style: "percent",
      maximumFractionDigits: 1,
    }).format(0.42);

    expect(within(dataOnlyRow!).getByText("Withheld")).toHaveAccessibleName(
      "Risk score withheld by assigned evidence view",
    );
    expect(within(dataOnlyRow!).queryByText(hiddenRisk)).not.toBeInTheDocument();
    expect(within(scoreOnlyRow!).getByText(visibleRisk)).toBeInTheDocument();
  });
});
