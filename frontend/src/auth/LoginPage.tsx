import { type FormEvent, useEffect, useRef, useState } from "react";
import { ArrowRight, LockKeyhole, Mail, ShieldCheck } from "lucide-react";
import { Navigate, useLocation, useNavigate } from "react-router-dom";
import { normalizeApiError } from "../lib/api";
import { useAuth } from "./AuthContext";
import "./auth.css";

interface LoginLocationState {
  from?: {
    pathname?: string;
    search?: string;
    hash?: string;
  };
}

function destinationFromState(state: unknown): string {
  if (!state || typeof state !== "object") return "/";
  const from = (state as LoginLocationState).from;
  if (!from?.pathname?.startsWith("/") || from.pathname.startsWith("//")) return "/";
  return `${from.pathname}${from.search ?? ""}${from.hash ?? ""}`;
}

export function LoginPage() {
  const { isAuthenticated, isLoading, login } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const emailRef = useRef<HTMLInputElement>(null);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    emailRef.current?.focus();
  }, []);

  if (isAuthenticated) {
    return <Navigate to={destinationFromState(location.state)} replace />;
  }

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (isSubmitting) return;
    setError(null);
    setIsSubmitting(true);
    try {
      await login(email, password);
      navigate(destinationFromState(location.state), { replace: true });
    } catch (reason) {
      setError(normalizeApiError(reason, "Unable to sign in. Check your credentials and try again."));
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <main className="login-page">
      <section className="login-brand-panel" aria-label="Sentinel AML">
        <div className="login-brand">
          <span className="brand-mark brand-mark-large" aria-hidden="true">
            <ShieldCheck size={30} strokeWidth={1.9} />
          </span>
          <span>
            <strong>Sentinel AML</strong>
            <small>Financial crime operations</small>
          </span>
        </div>
        <div className="login-message">
          <p className="eyebrow">Investigation workspace</p>
          <h1>Clear signals. Defensible decisions.</h1>
          <p>
            Review prioritized alerts, understand model evidence, and record every
            decision in one controlled analyst workflow.
          </p>
        </div>
        <p className="login-governance-note">
          <ShieldCheck size={17} aria-hidden="true" />
          Model output supports review. Final decisions always remain with an authorized analyst.
        </p>
      </section>

      <section className="login-form-panel">
        <div className="login-card">
          <div className="login-card-header">
            <p className="eyebrow">Secure access</p>
            <h2>Sign in to your workspace</h2>
            <p>Use your analyst credentials to continue.</p>
          </div>

          <form onSubmit={handleSubmit} noValidate>
            {error ? (
              <div className="form-alert" role="alert">
                {error}
              </div>
            ) : null}

            <label className="field-label" htmlFor="email">
              Work email
            </label>
            <div className="input-with-icon">
              <Mail size={18} aria-hidden="true" />
              <input
                ref={emailRef}
                id="email"
                name="email"
                type="email"
                autoComplete="username"
                placeholder="analyst@company.com"
                value={email}
                onChange={(event) => setEmail(event.target.value)}
                required
                disabled={isSubmitting || isLoading}
              />
            </div>

            <label className="field-label" htmlFor="password">
              Password
            </label>
            <div className="input-with-icon">
              <LockKeyhole size={18} aria-hidden="true" />
              <input
                id="password"
                name="password"
                type="password"
                autoComplete="current-password"
                placeholder="Enter your password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                required
                disabled={isSubmitting || isLoading}
              />
            </div>

            <button
              className="button button-primary login-submit"
              type="submit"
              disabled={isSubmitting || isLoading || !email.trim() || !password}
            >
              <span>{isSubmitting ? "Signing in…" : "Sign in securely"}</span>
              {!isSubmitting ? <ArrowRight size={18} aria-hidden="true" /> : null}
            </button>
          </form>

          <p className="login-support">
            Access is monitored and limited to authorized personnel. Contact your administrator if
            you cannot sign in.
          </p>
        </div>
      </section>
    </main>
  );
}
