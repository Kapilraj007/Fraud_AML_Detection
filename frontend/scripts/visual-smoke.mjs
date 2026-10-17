import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { chromium } from "playwright-core";

const baseUrl = process.env.VISUAL_BASE_URL || "http://127.0.0.1:4173";
const browserPath = findBrowser();
const outputDir = resolve(process.cwd(), "..", ".tmp", "frontend-visual");
mkdirSync(outputDir, { recursive: true });

const browser = await chromium.launch({ executablePath: browserPath, headless: true });
const errors = [];
const checks = [];

try {
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
  await context.addInitScript(() => {
    sessionStorage.setItem(
      "sentinel.auth.tokens",
      JSON.stringify({
        access_token: "visual-access-token",
        refresh_token: "visual-refresh-token",
        token_type: "bearer",
        expires_in: 900,
        refresh_expires_in: 604800,
      }),
    );
  });

  const page = await context.newPage();
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(`console: ${message.text()}`);
  });
  page.on("pageerror", (error) => errors.push(`page: ${error.message}`));
  await installMockApi(page);

  await inspect(page, "/", "desktop-overview", checks);
  await inspect(page, "/cases/case-001", "desktop-case", checks);
  await inspect(page, "/monitoring", "desktop-monitoring", checks);
  await inspect(page, "/transactions", "desktop-transactions", checks);
  await inspect(page, "/governance", "desktop-governance", checks);

  await page.setViewportSize({ width: 390, height: 844 });
  await inspect(page, "/cases", "mobile-queue", checks);
  await inspect(page, "/cases/case-001", "mobile-case", checks);
  await inspect(page, "/monitoring", "mobile-monitoring", checks);

  await context.close();
} finally {
  await browser.close();
}

const report = { browserPath, baseUrl, errors, checks };
writeFileSync(resolve(outputDir, "report.json"), JSON.stringify(report, null, 2));
console.log(JSON.stringify(report, null, 2));
if (errors.length || checks.some((check) => !check.passed)) process.exitCode = 1;

function findBrowser() {
  const candidates = [
    process.env.PLAYWRIGHT_BROWSER_PATH,
    "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
    "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
    "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
  ].filter(Boolean);
  const found = candidates.find((candidate) => existsSync(candidate));
  if (!found) throw new Error("No supported local Chromium browser was found");
  return found;
}

async function inspect(page, path, name, results) {
  await page.goto(`${baseUrl}${path}`, { waitUntil: "networkidle" });
  await page.locator("main").first().waitFor({ state: "visible" });
  await page.screenshot({ path: resolve(outputDir, `${name}.png`), fullPage: true });
  const layout = await page.evaluate(() => {
    const root = document.documentElement;
    const hasHorizontalOverflow = root.scrollWidth > root.clientWidth + 1;
    const sidebar = document.querySelector(".sidebar");
    const closedMobileSidebarExposed = Boolean(
      window.innerWidth <= 900 &&
        sidebar &&
        !sidebar.classList.contains("is-open") &&
        getComputedStyle(sidebar).visibility !== "hidden",
    );
    const emptyButtons = [...document.querySelectorAll("button")].filter((button) => {
      const name = button.getAttribute("aria-label") || button.textContent?.trim();
      return !name;
    }).length;
    const unlabeledInputs = [...document.querySelectorAll("input, select, textarea")].filter(
      (input) => {
        const id = input.getAttribute("id");
        return !input.getAttribute("aria-label") && !(id && document.querySelector(`label[for="${id}"]`)) && !input.closest("label");
      },
    ).length;
    return {
      title: document.title,
      hasHorizontalOverflow,
      closedMobileSidebarExposed,
      emptyButtons,
      unlabeledInputs,
      mainCount: document.querySelectorAll("main").length,
      navCount: document.querySelectorAll("nav").length,
      bodyWidth: root.clientWidth,
      scrollWidth: root.scrollWidth,
    };
  });
  results.push({
    name,
    path,
    passed:
      !layout.hasHorizontalOverflow &&
      !layout.closedMobileSidebarExposed &&
      layout.emptyButtons === 0 &&
      layout.unlabeledInputs === 0 &&
      layout.mainCount === 1 &&
      layout.navCount >= 1,
    ...layout,
  });
}

async function installMockApi(page) {
  const now = "2026-10-14T08:30:00Z";
  const cases = [
    makeCase("case-001", 1, 0.94, "in_review", "reliability_gated_case_context"),
    makeCase("case-002", 2, 0.88, "open", "generic_explanation"),
    makeCase("case-003", 3, 0.79, "claimed", "score_only"),
    makeCase("case-004", 4, 0.72, "open", "data_only"),
    makeCase("case-005", 5, 0.66, "more_info_requested", "score_only"),
    makeCase("case-006", 6, 0.58, "dismissed", "generic_explanation"),
  ];
  const featureSets = [
    { amount_log1p: { ks: 0.08, p_value: 0.21, drifted: false }, source_txn_count_1h: { ks: 0.11, p_value: 0.09, drifted: false }, new_counterparty: { ks: 0.05, p_value: 0.5, drifted: false } },
    { amount_log1p: { ks: 0.13, p_value: 0.08, drifted: false }, source_txn_count_1h: { ks: 0.18, p_value: 0.03, drifted: true }, new_counterparty: { ks: 0.09, p_value: 0.2, drifted: false } },
    { amount_log1p: { ks: 0.21, p_value: 0.01, drifted: true }, source_txn_count_1h: { ks: 0.27, p_value: 0.002, drifted: true }, new_counterparty: { ks: 0.16, p_value: 0.04, drifted: true } },
  ];
  const history = featureSets.map((feature_metrics, index) => ({
    id: `snapshot-${index}`,
    model_version_id: "model-001",
    window_start: `2026-10-14T0${5 + index}:00:00Z`,
    window_end: `2026-10-14T0${6 + index}:00:00Z`,
    status: index === 2 ? "drift_detected" : index === 1 ? "watch" : "stable",
    adwin_detected: index === 2,
    feature_metrics,
    prediction_metrics: { score_mean: 0.31 + index * 0.06, score_p95: 0.78 + index * 0.04 },
    created_at: now,
  }));

  await page.route("**/api/v1/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const path = url.pathname;
    let body;
    let status = 200;

    if (path.endsWith("/auth/me")) {
      body = { id: "user-001", email: "mira.chen@example.test", full_name: "Mira Chen", role: "admin", is_active: true, created_at: now, updated_at: now, last_login_at: now };
    } else if (path.endsWith("/auth/refresh")) {
      body = { access_token: "visual-access-token-2", refresh_token: "visual-refresh-token-2", token_type: "bearer", expires_in: 900, refresh_expires_in: 604800 };
    } else if (/\/cases\/[^/]+\/(claim|start-review)$/.test(path)) {
      body = { ...cases[0], status: path.endsWith("claim") ? "claimed" : "in_review" };
    } else if (/\/cases\/[^/]+\/disposition$/.test(path)) {
      status = 201;
      body = { id: "disposition-001", case_id: "case-001", analyst_id: "user-001", action: "escalate", confidence: 0.8, notes: "Escalated for enhanced review.", model_agreement: true, decision_seconds: 184, created_at: now };
    } else if (/\/cases\/[^/]+$/.test(path)) {
      const selected = cases.find((item) => path.endsWith(item.id)) || cases[0];
      body = {
        ...selected,
        transactions: [{ id: "transaction-001", event_id: "EVT-20261014-82941", dataset: "ibm_aml_hi_small", event_time: "2026-10-14T08:07:00Z", masked_source_account: "********4821", masked_destination_account: "********9074", amount: "48750.00", currency: "USD", transaction_type: "TRANSFER", created_at: now }],
        predictions: [{ id: "prediction-001", transaction_id: "transaction-001", model_version_id: "model-001", supervised_score: 0.91, anomaly_score: 0.83, combined_score: 0.94, decision_threshold: 0.62, predicted_positive: true, uncertainty: 0.12, reliability_score: 0.96, reliability_status: "reliable", explanation_gate_reason: null, drift_status: "watch", created_at: now, explanation: { top_features: [{ feature: "source_txn_count_1h", display_name: "one-hour transaction velocity", value: 0.44, direction: "increases_risk" }, { feature: "amount_to_source_mean_30d_ratio", display_name: "amount relative to account history", value: 0.31, direction: "increases_risk" }, { feature: "new_counterparty", display_name: "new counterparty activity", value: 0.18, direction: "increases_risk" }, { feature: "account_tenure_days", display_name: "account tenure", value: -0.12, direction: "decreases_risk" }, { feature: "destination_fan_in_24h", display_name: "inbound fan-in", value: 0.09, direction: "increases_risk" }], narrative: "Risk is driven primarily by elevated one-hour transaction velocity, an amount well above the account baseline, and new counterparty activity.", is_withheld: false, gate_reason: null } }],
        dispositions: [],
      };
    } else if (path.endsWith("/cases")) {
      body = cases;
    } else if (path.endsWith("/monitoring/status")) {
      body = { active_model_id: "model-001", active_model_version: "sprint2-lightgbm-v1", status: "drift_detected", latest_snapshot: history[2] };
    } else if (path.endsWith("/monitoring/history")) {
      body = history;
    } else if (path.endsWith("/monitoring/retraining-requests")) {
      status = 201;
      body = { id: "request-001", status: "open", reason: "Sustained drift across monitored features.", requested_by_id: "user-001", drift_snapshot_id: "snapshot-2", created_at: now, updated_at: now };
    } else if (path.endsWith("/models/active")) {
      body = { id: "model-001", version: "sprint2-lightgbm-v1", name: "AML primary", algorithm: "LightGBM", artifact_uri: "artifacts/models/sprint2-lightgbm", artifact_sha256: "a".repeat(64), feature_schema: { version: 1, feature_count: 48 }, metrics: { pr_auc: 0.1398, roc_auc: 0.9695, f1: 0.1954, case_recall_at_capacity: 0.768 }, status: "active", created_at: now, activated_at: "2026-10-13T18:45:00Z" };
    } else if (path.endsWith("/transactions")) {
      body = Array.from({ length: 12 }, (_, index) => ({ id: `transaction-${index}`, event_id: `EVT-20261014-${82941 - index}`, dataset: "ibm_aml_hi_small", event_time: `2026-10-14T08:${String(30 - index).padStart(2, "0")}:00Z`, masked_source_account: `********${4821 + index}`, masked_destination_account: `********${9074 - index}`, amount: String(48750 - index * 1875), currency: "USD", transaction_type: index % 3 === 0 ? "TRANSFER" : "PAYMENT", created_at: now }));
    } else if (path.endsWith("/audit/events")) {
      body = Array.from({ length: 8 }, (_, index) => ({ id: `audit-${index}`, actor_user_id: "user-001", event_type: index % 2 ? "case.review_started" : "case.claimed", resource_type: "case", resource_id: `case-00${index + 1}`, request_id: `request-${index}`, safe_details: { source: "analyst_console" }, created_at: `2026-10-14T08:${String(25 - index).padStart(2, "0")}:00Z` }));
    } else {
      status = 404;
      body = { detail: `No visual-smoke fixture for ${path}` };
    }
    await route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });
  });
}

function makeCase(id, rank, risk, status, studyArm) {
  return {
    id,
    review_id: `REV-20261014-${String(rank).padStart(4, "0")}`,
    status,
    queue_rank: rank,
    risk_score: risk,
    assigned_to_id: status === "open" ? null : "user-001",
    study_arm: studyArm,
    alert_hour: "2026-10-14T08:00:00Z",
    updated_at: "2026-10-14T08:25:00Z",
    created_at: "2026-10-14T08:10:00Z",
  };
}
