import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { BriefcaseBusiness, CircleAlert } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";

import { useAuth } from "../../auth/AuthContext";
import { ApiError, api, normalizeApiError } from "../../lib/api";
import type { CaseDispositionCreate, CaseStatus } from "../../lib/types";
import { CaseDetailPanel } from "./CaseDetailPanel";
import { CaseFilters, type CaseFilterValue } from "./CaseFilters";
import { CaseQueue } from "./CaseQueue";
import "./cases.css";

const QUEUE_REFRESH_MS = 30_000;

export function CaseWorkspace() {
  const { caseId } = useParams<{ caseId?: string }>();
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const queryClient = useQueryClient();
  const { user } = useAuth();
  const [staleMessage, setStaleMessage] = useState<string>();
  const [clock, setClock] = useState(() => Date.now());

  const filters = useMemo<CaseFilterValue>(() => ({
    search: searchParams.get("q") ?? "",
    status: parseStatus(searchParams.get("status")),
    assignedToMe: searchParams.get("mine") === "1",
  }), [searchParams]);

  const queueQuery = useQuery({
    queryKey: ["cases", "queue", filters.status, filters.assignedToMe],
    queryFn: () => api.cases.list({
      status: filters.status === "all" ? undefined : filters.status,
      assigned_to_me: filters.assignedToMe || undefined,
      limit: 200,
    }),
    refetchInterval: QUEUE_REFRESH_MS,
    refetchIntervalInBackground: false,
  });

  const visibleCases = useMemo(() => {
    const query = filters.search.trim().toLocaleLowerCase();
    const cases = queueQuery.data ?? [];
    if (!query) return cases;
    return cases.filter((item) =>
      item.review_id.toLocaleLowerCase().includes(query) ||
      item.id.toLocaleLowerCase().includes(query),
    );
  }, [filters.search, queueQuery.data]);

  const detailQuery = useQuery({
    queryKey: ["cases", "detail", caseId],
    queryFn: () => api.cases.get(caseId!),
    enabled: Boolean(caseId),
  });

  useEffect(() => {
    if (detailQuery.data?.status !== "in_review") return undefined;
    const timer = window.setInterval(() => setClock(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [detailQuery.data?.status]);

  const syncCaseData = async (id: string) => {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: ["cases", "queue"] }),
      queryClient.invalidateQueries({ queryKey: ["cases", "detail", id] }),
    ]);
  };

  const claimMutation = useMutation({
    mutationFn: (id: string) => api.cases.claim(id, "open"),
    onSuccess: (_, id) => {
      setStaleMessage(undefined);
      return syncCaseData(id);
    },
    onError: (error, id) => handleMutationError(error, id),
  });

  const reviewMutation = useMutation({
    mutationFn: ({ id, status }: { id: string; status: CaseStatus }) => api.cases.startReview(id, status),
    onSuccess: (_, { id }) => {
      setStaleMessage(undefined);
      saveReviewStart(id, Date.now());
      setClock(Date.now());
      return syncCaseData(id);
    },
    onError: (error, { id }) => handleMutationError(error, id),
  });

  const dispositionMutation = useMutation({
    mutationFn: ({ id, payload }: { id: string; payload: CaseDispositionCreate }) =>
      api.cases.disposition(id, payload),
    onSuccess: (_, { id }) => {
      setStaleMessage(undefined);
      clearReviewStart(id);
      return syncCaseData(id);
    },
    onError: (error, { id }) => handleMutationError(error, id),
  });

  async function handleMutationError(error: unknown, id: string) {
    if (isConflict(error)) {
      setStaleMessage("This case changed while you were reviewing it. The latest server state has been loaded; re-check it before continuing.");
      await syncCaseData(id);
    }
  }

  function updateFilters(next: CaseFilterValue) {
    const params = new URLSearchParams();
    if (next.search.trim()) params.set("q", next.search.trim());
    if (next.status !== "all") params.set("status", next.status);
    if (next.assignedToMe) params.set("mine", "1");
    setSearchParams(params, { replace: true });
  }

  function selectCase(id: string) {
    setStaleMessage(undefined);
    navigate(caseHref(id, searchParams));
  }

  function backToQueue() {
    navigate(`/cases${searchParams.size ? `?${searchParams.toString()}` : ""}`);
  }

  const activeDetail = detailQuery.data;
  const reviewStartedAt = activeDetail
    ? getReviewStart(activeDetail.id) ?? Date.parse(activeDetail.updated_at)
    : clock;
  const elapsedSeconds = activeDetail?.status === "in_review"
    ? Math.max(0, Math.floor((clock - reviewStartedAt) / 1000))
    : 0;
  const actionPending = claimMutation.isPending || reviewMutation.isPending || dispositionMutation.isPending;

  return (
    <div className="case-workspace">
      <header className="case-workspace__header">
        <div>
          <p className="case-workspace__eyebrow"><BriefcaseBusiness aria-hidden="true" size={16} /> Investigation workspace</p>
          <h1>Case review</h1>
          <p>Prioritize alerts, examine evidence, and record a defensible decision.</p>
        </div>
        <div className="case-workspace__summary" aria-label="Queue summary">
          <span><strong>{queueQuery.data?.length ?? 0}</strong><small>Loaded cases</small></span>
          <span><strong>{queueQuery.data?.filter((item) => item.status === "open").length ?? 0}</strong><small>Unclaimed</small></span>
          <span><strong>{queueQuery.data?.filter((item) => item.status === "in_review").length ?? 0}</strong><small>In review</small></span>
        </div>
      </header>

      <CaseFilters value={filters} resultCount={visibleCases.length} onChange={updateFilters} />

      <div className="case-workspace__layout">
        <aside className={`case-workspace__queue${caseId ? " case-workspace__queue--case-selected" : ""}`}>
          <CaseQueue
            cases={visibleCases}
            selectedCaseId={caseId}
            isLoading={queueQuery.isLoading}
            isRefreshing={queueQuery.isFetching}
            error={!queueQuery.data && queueQuery.error ? normalizeApiError(queueQuery.error) : undefined}
            lastUpdatedAt={queueQuery.dataUpdatedAt}
            onSelect={selectCase}
            onRefresh={() => void queueQuery.refetch()}
          />
        </aside>

        <section className={`case-workspace__detail${caseId ? " case-workspace__detail--visible" : ""}`}>
          {claimMutation.error && !isConflict(claimMutation.error) ? (
            <ActionError error={claimMutation.error} />
          ) : null}
          {reviewMutation.error && !isConflict(reviewMutation.error) ? (
            <ActionError error={reviewMutation.error} />
          ) : null}
          <CaseDetailPanel
            caseDetail={activeDetail}
            user={user}
            isLoading={Boolean(caseId) && detailQuery.isLoading}
            error={detailQuery.error ? normalizeApiError(detailQuery.error) : undefined}
            staleMessage={staleMessage}
            elapsedSeconds={elapsedSeconds}
            actionPending={actionPending}
            onBack={backToQueue}
            onReload={() => {
              setStaleMessage(undefined);
              void detailQuery.refetch();
            }}
            onClaim={async () => {
              if (caseId) await claimMutation.mutateAsync(caseId);
            }}
            onStartReview={async () => {
              if (caseId && activeDetail) {
                await reviewMutation.mutateAsync({ id: caseId, status: activeDetail.status });
              }
            }}
            onDisposition={async (payload) => {
              if (caseId) await dispositionMutation.mutateAsync({ id: caseId, payload });
            }}
          />
        </section>
      </div>
    </div>
  );
}

function ActionError({ error }: { error: unknown }) {
  return (
    <div className="case-action-error" role="alert">
      <CircleAlert aria-hidden="true" size={18} /> {normalizeApiError(error)}
    </div>
  );
}

function parseStatus(value: string | null): CaseStatus | "all" {
  const statuses: CaseStatus[] = [
    "open",
    "claimed",
    "in_review",
    "escalated",
    "dismissed",
    "more_info_requested",
    "closed",
  ];
  return value && statuses.includes(value as CaseStatus) ? value as CaseStatus : "all";
}

function caseHref(id: string, searchParams: URLSearchParams) {
  return `/cases/${id}${searchParams.size ? `?${searchParams.toString()}` : ""}`;
}

function isConflict(error: unknown) {
  return error instanceof ApiError && error.status === 409;
}

function storageKey(id: string) {
  return `aml-review-start:${id}`;
}

function getReviewStart(id: string) {
  try {
    const value = window.sessionStorage.getItem(storageKey(id));
    const parsed = value ? Number(value) : Number.NaN;
    return Number.isFinite(parsed) ? parsed : undefined;
  } catch {
    return undefined;
  }
}

function saveReviewStart(id: string, startedAt: number) {
  try {
    window.sessionStorage.setItem(storageKey(id), String(startedAt));
  } catch {
    // Review timing still works in-memory when browser storage is unavailable.
  }
}

function clearReviewStart(id: string) {
  try {
    window.sessionStorage.removeItem(storageKey(id));
  } catch {
    // A stale session value is harmless once the case leaves review.
  }
}
