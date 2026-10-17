import type {
  AuditEvent,
  AuditListParams,
  Case,
  CaseDetail,
  CaseDisposition,
  CaseDispositionCreate,
  CaseListParams,
  CaseStatus,
  DriftSnapshot,
  LoginRequest,
  ModelPromotionRequest,
  ModelVersion,
  MonitoringHistoryParams,
  MonitoringStatus,
  Prediction,
  PredictionRequest,
  RetrainingRequest,
  RetrainingRequestCreate,
  TokenResponse,
  Transaction,
  TransactionCreate,
  TransactionListParams,
  User,
  ValidationIssue,
} from "./types";

const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim();

export const API_BASE_URL = (configuredBaseUrl || "/api/v1").replace(/\/+$/, "");
export const TOKEN_STORAGE_KEY = "sentinel.auth.tokens";
export const AUTH_SESSION_EXPIRED_EVENT = "sentinel:auth-session-expired";

export class ApiError extends Error {
  readonly status: number;
  readonly requestId: string | null;
  readonly issues: ValidationIssue[];

  constructor(
    message: string,
    options: {
      status?: number;
      requestId?: string | null;
      issues?: ValidationIssue[];
      cause?: unknown;
    } = {},
  ) {
    super(message, { cause: options.cause });
    this.name = "ApiError";
    this.status = options.status ?? 0;
    this.requestId = options.requestId ?? null;
    this.issues = options.issues ?? [];
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isValidationIssue(value: unknown): value is ValidationIssue {
  return isRecord(value) && (typeof value.msg === "string" || Array.isArray(value.loc));
}

function formatValidationIssue(issue: ValidationIssue): string {
  const location = issue.loc
    ?.filter((part) => part !== "body" && part !== "query" && part !== "path")
    .join(".");
  const message = issue.msg?.trim() || "Invalid value";
  return location ? `${location}: ${message}` : message;
}

export function normalizeApiError(
  error: unknown,
  fallback = "Something went wrong. Please try again.",
): string {
  if (error instanceof ApiError) return error.message || fallback;
  if (error instanceof Error) return error.message || fallback;
  if (typeof error === "string") return error.trim() || fallback;

  if (isRecord(error)) {
    if (typeof error.detail === "string") return error.detail.trim() || fallback;
    if (typeof error.message === "string") return error.message.trim() || fallback;
    if (Array.isArray(error.detail)) {
      const messages = error.detail
        .filter(isValidationIssue)
        .map(formatValidationIssue)
        .filter(Boolean);
      if (messages.length > 0) return messages.join("; ");
    }
  }

  return fallback;
}

function isTokenResponse(value: unknown): value is TokenResponse {
  if (!isRecord(value)) return false;
  return (
    typeof value.access_token === "string" &&
    value.access_token.length > 0 &&
    typeof value.refresh_token === "string" &&
    value.refresh_token.length > 0 &&
    value.token_type === "bearer" &&
    typeof value.expires_in === "number" &&
    typeof value.refresh_expires_in === "number"
  );
}

let memoryTokens: TokenResponse | null = null;

export const tokenStore = {
  get(): TokenResponse | null {
    if (typeof window === "undefined") return memoryTokens;
    try {
      const raw = window.sessionStorage.getItem(TOKEN_STORAGE_KEY);
      if (!raw) return memoryTokens;
      const parsed: unknown = JSON.parse(raw);
      if (isTokenResponse(parsed)) {
        memoryTokens = parsed;
        return parsed;
      }
      window.sessionStorage.removeItem(TOKEN_STORAGE_KEY);
    } catch {
      // A memory fallback keeps the current tab usable if storage is blocked.
    }
    return memoryTokens;
  },

  set(tokens: TokenResponse): void {
    memoryTokens = tokens;
    if (typeof window === "undefined") return;
    try {
      // Access and rotating refresh tokens are replaced in a single write.
      window.sessionStorage.setItem(TOKEN_STORAGE_KEY, JSON.stringify(tokens));
    } catch {
      // A memory fallback is already populated.
    }
  },

  clear(): void {
    memoryTokens = null;
    if (typeof window === "undefined") return;
    try {
      window.sessionStorage.removeItem(TOKEN_STORAGE_KEY);
    } catch {
      // Clearing the memory copy is sufficient when storage is unavailable.
    }
  },
};

function emitSessionExpired(): void {
  if (typeof window !== "undefined") {
    window.dispatchEvent(new Event(AUTH_SESSION_EXPIRED_EVENT));
  }
}

function expireSession(): void {
  tokenStore.clear();
  emitSessionExpired();
}

async function responsePayload(response: Response): Promise<unknown> {
  const contentType = response.headers.get("content-type") ?? "";
  if (contentType.includes("json")) {
    try {
      return await response.json();
    } catch {
      return null;
    }
  }
  try {
    const text = await response.text();
    return text || null;
  } catch {
    return null;
  }
}

export async function errorFromResponse(response: Response): Promise<ApiError> {
  const payload = await responsePayload(response);
  const detail = isRecord(payload) ? payload.detail : undefined;
  const issues = Array.isArray(detail) ? detail.filter(isValidationIssue) : [];
  const fallback = response.statusText || `Request failed (${response.status})`;
  const message = normalizeApiError(payload, fallback);
  return new ApiError(message, {
    status: response.status,
    requestId: response.headers.get("x-request-id"),
    issues,
  });
}

let refreshPromise: Promise<TokenResponse> | null = null;

async function performRefresh(): Promise<TokenResponse> {
  const current = tokenStore.get();
  if (!current?.refresh_token) {
    throw new ApiError("Your session has expired. Please sign in again.", { status: 401 });
  }

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/auth/refresh`, {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ refresh_token: current.refresh_token }),
    });
  } catch (cause) {
    throw new ApiError("Unable to refresh your session. Check your connection and sign in again.", {
      cause,
    });
  }

  if (!response.ok) throw await errorFromResponse(response);
  const payload: unknown = await responsePayload(response);
  if (!isTokenResponse(payload)) {
    throw new ApiError("The authentication service returned an invalid token response.", {
      status: response.status,
    });
  }
  tokenStore.set(payload);
  return payload;
}

export function refreshSession(): Promise<TokenResponse> {
  if (refreshPromise) return refreshPromise;

  refreshPromise = performRefresh()
    .catch((error: unknown) => {
      expireSession();
      throw error;
    })
    .finally(() => {
      refreshPromise = null;
    });
  return refreshPromise;
}

interface ApiRequestOptions extends Omit<RequestInit, "body"> {
  body?: unknown;
  auth?: boolean;
  retryUnauthorized?: boolean;
}

function requestBody(value: unknown): BodyInit | undefined {
  if (value == null) return undefined;
  if (
    typeof value === "string" ||
    value instanceof Blob ||
    value instanceof FormData ||
    value instanceof URLSearchParams ||
    value instanceof ArrayBuffer
  ) {
    return value;
  }
  return JSON.stringify(value);
}

async function requestOnce(
  path: string,
  options: ApiRequestOptions,
  accessToken: string | null,
): Promise<Response> {
  const { auth, retryUnauthorized, ...requestOptions } = options;
  void auth;
  void retryUnauthorized;
  const headers = new Headers(options.headers);
  headers.set("Accept", "application/json");
  const body = requestBody(options.body);
  if (body && !(body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  if (accessToken) headers.set("Authorization", `Bearer ${accessToken}`);

  try {
    return await fetch(`${API_BASE_URL}${path.startsWith("/") ? path : `/${path}`}`, {
      ...requestOptions,
      body,
      headers,
    });
  } catch (cause) {
    if (cause instanceof DOMException && cause.name === "AbortError") throw cause;
    throw new ApiError("Unable to reach the service. Check your connection and try again.", {
      cause,
    });
  }
}

export async function apiRequest<T>(path: string, options: ApiRequestOptions = {}): Promise<T> {
  const { auth = true, retryUnauthorized = true } = options;
  const tokenUsed = auth ? tokenStore.get()?.access_token ?? null : null;
  let response = await requestOnce(path, options, tokenUsed);

  if (auth && response.status === 401 && retryUnauthorized) {
    const latestToken = tokenStore.get()?.access_token ?? null;
    if (!latestToken || latestToken === tokenUsed) {
      await refreshSession();
    }
    const replacementToken = tokenStore.get()?.access_token ?? null;
    if (!replacementToken) {
      expireSession();
      throw new ApiError("Your session has expired. Please sign in again.", { status: 401 });
    }
    response = await requestOnce(path, { ...options, retryUnauthorized: false }, replacementToken);
    if (response.status === 401) expireSession();
  }

  if (!response.ok) throw await errorFromResponse(response);
  if (response.status === 204) return undefined as T;
  return (await responsePayload(response)) as T;
}

function queryString(params: object): string {
  const search = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") {
      search.set(key, String(value));
    }
  });
  const encoded = search.toString();
  return encoded ? `?${encoded}` : "";
}

async function login(credentials: LoginRequest): Promise<TokenResponse> {
  const tokens = await apiRequest<TokenResponse>("/auth/token", {
    method: "POST",
    body: credentials,
    auth: false,
    retryUnauthorized: false,
  });
  tokenStore.set(tokens);
  return tokens;
}

async function logout(): Promise<void> {
  const refreshToken = tokenStore.get()?.refresh_token;
  try {
    if (refreshToken) {
      await apiRequest<{ message: string }>("/auth/logout", {
        method: "POST",
        body: { refresh_token: refreshToken },
        retryUnauthorized: false,
      });
    }
  } finally {
    tokenStore.clear();
  }
}

const getCurrentUser = (): Promise<User> => apiRequest<User>("/auth/me");

const getCases = (params: CaseListParams = {}): Promise<Case[]> =>
  apiRequest<Case[]>(`/cases${queryString(params)}`);

const getCase = (caseId: string): Promise<CaseDetail> =>
  apiRequest<CaseDetail>(`/cases/${encodeURIComponent(caseId)}`);

const claimCase = (caseId: string, expectedStatus: CaseStatus = "open"): Promise<Case> =>
  apiRequest<Case>(`/cases/${encodeURIComponent(caseId)}/claim`, {
    method: "POST",
    body: { expected_status: expectedStatus },
  });

const startReview = (
  caseId: string,
  expectedStatus: CaseStatus = "claimed",
): Promise<Case> =>
  apiRequest<Case>(`/cases/${encodeURIComponent(caseId)}/start-review`, {
    method: "POST",
    body: { expected_status: expectedStatus },
  });

const disposition = (
  caseId: string,
  payload: CaseDispositionCreate,
): Promise<CaseDisposition> =>
  apiRequest<CaseDisposition>(`/cases/${encodeURIComponent(caseId)}/disposition`, {
    method: "POST",
    body: payload,
  });

const materializeCases = (alertHour: string, capacity = 25): Promise<Case[]> =>
  apiRequest<Case[]>("/cases/materialize", {
    method: "POST",
    body: { alert_hour: alertHour, capacity },
  });

const getTransactions = (params: TransactionListParams = {}): Promise<Transaction[]> =>
  apiRequest<Transaction[]>(`/transactions${queryString(params)}`);

const getTransaction = (transactionId: string): Promise<Transaction> =>
  apiRequest<Transaction>(`/transactions/${encodeURIComponent(transactionId)}`);

const createTransaction = (payload: TransactionCreate): Promise<Transaction> =>
  apiRequest<Transaction>("/transactions", { method: "POST", body: payload });

const scorePrediction = (payload: PredictionRequest): Promise<Prediction> =>
  apiRequest<Prediction>("/predictions/score", { method: "POST", body: payload });

const scorePredictionBatch = (rows: PredictionRequest[]): Promise<Prediction[]> =>
  apiRequest<Prediction[]>("/predictions/batch-score", {
    method: "POST",
    body: { rows },
  });

const getMonitoringStatus = (): Promise<MonitoringStatus> =>
  apiRequest<MonitoringStatus>("/monitoring/status");

const getMonitoringHistory = (
  params: MonitoringHistoryParams = {},
): Promise<DriftSnapshot[]> =>
  apiRequest<DriftSnapshot[]>(`/monitoring/history${queryString(params)}`);

const requestRetraining = (payload: RetrainingRequestCreate): Promise<RetrainingRequest> =>
  apiRequest<RetrainingRequest>("/monitoring/retraining-requests", {
    method: "POST",
    body: payload,
  });

const getActiveModel = (): Promise<ModelVersion> => apiRequest<ModelVersion>("/models/active");

const promoteModel = (modelId: string, payload: ModelPromotionRequest): Promise<ModelVersion> =>
  apiRequest<ModelVersion>(`/models/${encodeURIComponent(modelId)}/promote`, {
    method: "POST",
    body: payload,
  });

const getAuditEvents = (params: AuditListParams = {}): Promise<AuditEvent[]> =>
  apiRequest<AuditEvent[]>(`/audit/events${queryString(params)}`);

export const api = {
  login,
  logout,
  getCurrentUser,
  getCases,
  getCase,
  claimCase,
  startReview,
  disposition,
  materializeCases,
  getTransactions,
  getTransaction,
  createTransaction,
  scorePrediction,
  scorePredictionBatch,
  getMonitoringStatus,
  getMonitoringHistory,
  requestRetraining,
  getActiveModel,
  promoteModel,
  getAuditEvents,
  auth: {
    login,
    logout,
    me: getCurrentUser,
  },
  cases: {
    list: getCases,
    get: getCase,
    claim: claimCase,
    startReview,
    disposition,
    materialize: materializeCases,
  },
  transactions: {
    list: getTransactions,
    get: getTransaction,
    create: createTransaction,
  },
  predictions: {
    score: scorePrediction,
    scoreBatch: scorePredictionBatch,
  },
  monitoring: {
    status: getMonitoringStatus,
    history: getMonitoringHistory,
    requestRetraining,
  },
  models: {
    active: getActiveModel,
    promote: promoteModel,
  },
  audit: {
    list: getAuditEvents,
  },
};
