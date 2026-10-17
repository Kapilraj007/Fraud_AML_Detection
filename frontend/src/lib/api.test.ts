import { afterEach, describe, expect, it, vi } from "vitest";
import {
  API_BASE_URL,
  ApiError,
  apiRequest,
  errorFromResponse,
  normalizeApiError,
  tokenStore,
} from "./api";
import type { TokenResponse } from "./types";

const oldTokens: TokenResponse = {
  access_token: "old-access",
  refresh_token: "old-refresh",
  token_type: "bearer",
  expires_in: 900,
  refresh_expires_in: 604_800,
};

const rotatedTokens: TokenResponse = {
  access_token: "new-access",
  refresh_token: "new-refresh",
  token_type: "bearer",
  expires_in: 900,
  refresh_expires_in: 604_800,
};

function jsonResponse(
  body: unknown,
  status = 200,
  headers: Record<string, string> = {},
): Response {
  return new Response(JSON.stringify(body), {
    status,
    statusText: status === 200 ? "OK" : "Request failed",
    headers: { "content-type": "application/json", ...headers },
  });
}

afterEach(() => {
  tokenStore.clear();
  vi.unstubAllGlobals();
});

describe("API error normalization", () => {
  it("turns FastAPI validation details into a concise field message", async () => {
    const response = jsonResponse(
      {
        detail: [
          { loc: ["body", "email"], msg: "Field required", type: "missing" },
          { loc: ["body", "password"], msg: "Too short", type: "value_error" },
        ],
      },
      422,
      { "x-request-id": "request-123" },
    );

    const error = await errorFromResponse(response);

    expect(error).toBeInstanceOf(ApiError);
    expect(error.status).toBe(422);
    expect(error.requestId).toBe("request-123");
    expect(error.issues).toHaveLength(2);
    expect(normalizeApiError(error)).toBe("email: Field required; password: Too short");
  });

  it("uses safe fallbacks for malformed and network errors", () => {
    expect(normalizeApiError({ detail: "Case was already claimed" })).toBe(
      "Case was already claimed",
    );
    expect(normalizeApiError(null, "Try again later")).toBe("Try again later");
    expect(normalizeApiError(new TypeError("Network request failed"))).toBe(
      "Network request failed",
    );
  });
});

describe("rotating token refresh", () => {
  it("shares one refresh across parallel 401 responses and retries both once", async () => {
    tokenStore.set(oldTokens);
    let releaseRefresh: ((response: Response) => void) | undefined;
    const refreshResponse = new Promise<Response>((resolve) => {
      releaseRefresh = resolve;
    });

    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const headers = new Headers(init?.headers);
      if (url === `${API_BASE_URL}/auth/refresh`) return refreshResponse;
      if (headers.get("authorization") === "Bearer old-access") {
        return Promise.resolve(jsonResponse({ detail: "expired" }, 401));
      }
      return Promise.resolve(jsonResponse([{ id: "case-1" }]));
    });
    vi.stubGlobal("fetch", fetchMock);

    const first = apiRequest<Array<{ id: string }>>("/cases");
    const second = apiRequest<Array<{ id: string }>>("/cases?assigned_to_me=true");

    await vi.waitFor(() => {
      expect(
        fetchMock.mock.calls.filter(([input]) => String(input).endsWith("/auth/refresh")),
      ).toHaveLength(1);
    });
    releaseRefresh?.(jsonResponse(rotatedTokens));

    await expect(Promise.all([first, second])).resolves.toEqual([
      [{ id: "case-1" }],
      [{ id: "case-1" }],
    ]);
    expect(tokenStore.get()).toEqual(rotatedTokens);

    const retriedRequests = fetchMock.mock.calls.filter(([, init]) =>
      new Headers(init?.headers).get("authorization")?.includes("new-access"),
    );
    expect(retriedRequests).toHaveLength(2);
  });

  it("clears the entire session when the shared refresh fails", async () => {
    tokenStore.set(oldTokens);
    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      if (String(input).endsWith("/auth/refresh")) {
        return Promise.resolve(jsonResponse({ detail: "refresh token is invalid or expired" }, 401));
      }
      return Promise.resolve(jsonResponse({ detail: "expired" }, 401));
    });
    vi.stubGlobal("fetch", fetchMock);

    const requests = [apiRequest("/cases"), apiRequest("/monitoring/status")];
    const results = await Promise.allSettled(requests);

    expect(results.every((result) => result.status === "rejected")).toBe(true);
    expect(tokenStore.get()).toBeNull();
    expect(
      fetchMock.mock.calls.filter(([input]) => String(input).endsWith("/auth/refresh")),
    ).toHaveLength(1);
  });
});
