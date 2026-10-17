# Sprint 3 Frontend Architecture

The Sprint 3 frontend is a typed React application for the existing FastAPI service. It presents one authenticated operations workspace with persistent sidebar navigation, rather than a collection of disconnected demo screens.

## Analyst workflow

1. Sign in with a short-lived access token and rotating refresh token.
2. Open the ranked investigation queue and filter the currently loaded API view.
3. Select a case, claim it, and start or resume review.
4. Inspect masked transaction details and the evidence permitted by the case's fixed study arm.
5. Record an escalation, dismissal, or request for more information with confidence, rationale, model agreement, and elapsed review time.
6. Review feature-level drift, ADWIN state, and active-model health. Senior investigators and administrators may open a human-gated retraining request.

Case mutations wait for the server response. A `409 Conflict` is treated as stale case state: the frontend reloads the queue and case detail instead of overwriting another analyst's work.

## Workspace routes

| Route | Purpose | Access |
| --- | --- | --- |
| `/` | Current-view queue metrics, model health, drift state, and workflow orientation | All authenticated roles |
| `/cases/:caseId?` | Auto-refreshing queue and complete review workflow | All authenticated roles |
| `/monitoring` | KS trend history, ADWIN state, and retraining request | All roles; request action is senior/admin only |
| `/transactions` | Masked transaction ledger with date filters and API pagination | All authenticated roles |
| `/governance` | Active model details and sanitized audit history | Admin only |

List endpoints return pages without a total count. The interface therefore labels figures as the current or loaded API view and does not present them as institution-wide totals.

## Study presentation

The backend assigns one immutable study arm per case. The frontend does not offer an analyst-controlled toggle because switching evidence during review would invalidate the evaluation.

| Study arm | Analyst evidence |
| --- | --- |
| `data_only` | Masked transaction and case context |
| `score_only` | Transaction context plus supervised, anomaly, and combined risk signals |
| `generic_explanation` | Scores plus the top signed feature contributions and narrative |
| `reliability_gated_case_context` | Scores and explanation only when reliability checks pass |

Unknown study-arm values use the conservative data-only presentation.

## Authentication and API behavior

Tokens are kept in session storage, limiting persistence to the current browser tab. Concurrent `401` responses share one in-flight refresh request so a rotating refresh token cannot be reused accidentally. Both tokens are replaced atomically; a failed refresh clears the session.

The Vite development server proxies `/api` and `/health` to `http://127.0.0.1:8000`. Production hosting may set `VITE_API_BASE_URL` to a complete `/api/v1` base URL.

## Verification

Run from `frontend/`:

```powershell
npm install
npm run lint
npm run typecheck
npm test
npm run build
```

For a browser-level responsive smoke after the build, keep `npm run preview`
running in one terminal and run `npm run visual:smoke` in another. The smoke
uses local fixtures, checks desktop and mobile routes for overflow and basic
labeling problems, and writes screenshots plus `report.json` under
`.tmp/frontend-visual/`.

The frontend checks complement, rather than replace, the repository's Python test suite.
