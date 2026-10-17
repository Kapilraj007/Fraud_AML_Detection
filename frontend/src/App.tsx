import { lazy, Suspense } from "react";
import { Navigate, Outlet, Route, Routes } from "react-router-dom";
import { ArrowLeft, ShieldAlert } from "lucide-react";
import { LoginPage } from "./auth/LoginPage";
import { ProtectedRoute } from "./auth/ProtectedRoute";
import { useAuth } from "./auth/AuthContext";
import { AppShell } from "./components/layout/AppShell";

const OverviewPage = lazy(() =>
  import("./features/dashboard").then((module) => ({ default: module.OverviewPage })),
);
const CaseWorkspace = lazy(() =>
  import("./features/cases").then((module) => ({ default: module.CaseWorkspace })),
);
const MonitoringPage = lazy(() =>
  import("./features/monitoring").then((module) => ({ default: module.MonitoringPage })),
);
const TransactionsPage = lazy(() =>
  import("./features/transactions").then((module) => ({ default: module.TransactionsPage })),
);
const GovernancePage = lazy(() =>
  import("./features/governance").then((module) => ({ default: module.GovernancePage })),
);

function AdminRoute() {
  const { user } = useAuth();
  return user?.role === "admin" ? <Outlet /> : <Navigate to="/" replace />;
}

function NotFoundPage() {
  return (
    <section className="not-found" aria-labelledby="not-found-title">
      <span className="not-found-icon" aria-hidden="true"><ShieldAlert size={26} /></span>
      <p className="eyebrow">Navigation error</p>
      <h2 id="not-found-title">This workspace view does not exist</h2>
      <p>The address may be outdated, or your role may not have access to the requested view.</p>
      <a className="button button-primary" href="/">
        <ArrowLeft size={16} aria-hidden="true" /> Return to command center
      </a>
    </section>
  );
}

export default function App() {
  return (
    <Suspense
      fallback={(
        <div className="route-loader" role="status" aria-live="polite">
          <span className="spinner" aria-hidden="true" />
          <span>Loading workspace view...</span>
        </div>
      )}
    >
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route element={<ProtectedRoute />}>
          <Route element={<AppShell />}>
            <Route index element={<OverviewPage />} />
            <Route path="cases" element={<CaseWorkspace />} />
            <Route path="cases/:caseId" element={<CaseWorkspace />} />
            <Route path="monitoring" element={<MonitoringPage />} />
            <Route path="transactions" element={<TransactionsPage />} />
            <Route element={<AdminRoute />}>
              <Route path="governance" element={<GovernancePage />} />
            </Route>
            <Route path="*" element={<NotFoundPage />} />
          </Route>
        </Route>
      </Routes>
    </Suspense>
  );
}
