import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { tokenStore } from "../lib/api";
import type { TokenResponse, User } from "../lib/types";
import { AuthProvider, useAuth } from "./AuthContext";
import { LoginPage } from "./LoginPage";
import { ProtectedRoute } from "./ProtectedRoute";

const tokens: TokenResponse = {
  access_token: "access-token",
  refresh_token: "refresh-token",
  token_type: "bearer",
  expires_in: 900,
  refresh_expires_in: 604_800,
};

const analyst: User = {
  id: "user-1",
  email: "analyst@example.com",
  full_name: "Ada Analyst",
  role: "fraud_analyst",
  is_active: true,
  created_at: "2026-10-14T08:00:00Z",
  updated_at: "2026-10-14T08:00:00Z",
  last_login_at: "2026-10-14T08:00:00Z",
};

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function Identity() {
  const { user } = useAuth();
  return <p>{user?.full_name}</p>;
}

afterEach(() => {
  tokenStore.clear();
  vi.unstubAllGlobals();
});

describe("authentication flow", () => {
  it("sends an unauthenticated deep link to login and preserves its destination", async () => {
    render(
      <MemoryRouter initialEntries={["/cases/case-42"]}>
        <AuthProvider>
          <Routes>
            <Route path="/login" element={<p>Sign-in route</p>} />
            <Route element={<ProtectedRoute />}>
              <Route path="/cases/:caseId" element={<p>Case detail</p>} />
            </Route>
          </Routes>
        </AuthProvider>
      </MemoryRouter>,
    );

    expect(await screen.findByText("Sign-in route")).toBeInTheDocument();
    expect(screen.queryByText("Case detail")).not.toBeInTheDocument();
  });

  it("restores a user from session-scoped tokens before revealing a protected route", async () => {
    tokenStore.set(tokens);
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(analyst)));

    render(
      <MemoryRouter initialEntries={["/cases"]}>
        <AuthProvider>
          <Routes>
            <Route path="/login" element={<p>Sign-in route</p>} />
            <Route element={<ProtectedRoute />}>
              <Route path="/cases" element={<Identity />} />
            </Route>
          </Routes>
        </AuthProvider>
      </MemoryRouter>,
    );

    expect(screen.getByRole("status")).toHaveTextContent("Restoring secure session");
    expect(await screen.findByText("Ada Analyst")).toBeInTheDocument();
    expect(screen.queryByText("Sign-in route")).not.toBeInTheDocument();
  });

  it("signs in and returns the analyst to the originally requested route", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse(tokens))
      .mockResolvedValueOnce(jsonResponse(analyst));
    vi.stubGlobal("fetch", fetchMock);
    render(
      <MemoryRouter
        initialEntries={[
          {
            pathname: "/login",
            state: { from: { pathname: "/cases/case-42", search: "?tab=evidence" } },
          },
        ]}
      >
        <AuthProvider>
          <Routes>
            <Route path="/login" element={<LoginPage />} />
            <Route element={<ProtectedRoute />}>
              <Route path="/cases/:caseId" element={<Identity />} />
            </Route>
          </Routes>
        </AuthProvider>
      </MemoryRouter>,
    );

    fireEvent.change(screen.getByLabelText("Work email"), {
      target: { value: "analyst@example.com" },
    });
    fireEvent.change(screen.getByLabelText("Password"), {
      target: { value: "correct horse battery staple" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Sign in securely" }));

    expect(await screen.findByText("Ada Analyst")).toBeInTheDocument();
    expect(tokenStore.get()).toEqual(tokens);
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
  });
});
