# Capacity-aware AML triage

This repository implements the data, modelling, operational backend, and
analyst console for a leakage-safe anti-money-laundering triage system. IBM
AMLworld HI-Small is used as governed research data; FastAPI provides the
workflow API; React provides the human-review workspace; Neon PostgreSQL stores
operational state. The project runs natively and does not use Docker.

## Status

- **Sprint 1:** complete as an implementation and bounded acceptance milestone.
  The independent `dev-100000` verifier passes, but that development matrix is
  not valid for model evaluation because its chronological test split has zero
  positives. See the [Sprint 1 audit](docs/sprint1_audit.md).
- **Sprint 2 backend:** LightGBM training, chronological evaluation,
  calibration, Isolation Forest, native TreeSHAP explanations, versioned model
  bundles, async persistence, migrations, authentication, RBAC, audit trails,
  prediction APIs, hourly capacity queues, case workflow, and drift/retraining
  services are implemented.
- **Sprint 3 frontend:** the typed React analyst console is complete, including
  secure sign-in, the ranked queue-to-disposition workflow, fixed study-arm
  evidence, accessible explanation visuals, drift monitoring, governed
  retraining requests, transactions, and admin-only model/audit views.
- **Database:** production configuration targets a pooled Neon endpoint with
  TLS. SQLite is used only by isolated automated tests. A live Neon migration
  requires a user-provided connection string; no credentials are committed.

The supplied [implementation plan](Part_A_Fraud_AML_Detection_SKI027_Implementation_Plan.docx)
is treated as project context. The implemented study contract deliberately
uses IBM AMLworld instead of the plan's proposed IEEE-CIS/PaySim combination,
and the user's Neon/no-Docker requirements override its infrastructure example.

## Backend scope

The selected primary classifier is CPU LightGBM. It is a good fit for the
imbalanced tabular workload, supports deterministic training and efficient
native contribution values, and can be stored without Python pickle. The
pipeline also fits a portable logistic-regression baseline and a secondary
Isolation Forest anomaly signal.

Key safeguards:

- exact chronological timestamp-group splits, with equal timestamps kept in
  one split;
- all positives retained and deterministic negative sampling confined to the
  training period;
- inverse-inclusion weighting for the sampled training negatives;
- full, unsampled validation and held-out test scoring;
- Platt calibration and threshold selection on validation only;
- evaluation labels, laundering case IDs, and typologies prohibited from
  serving inputs and operational tables;
- SHA-256 verification of every model-bundle payload before loading; and
- deterministic top-*N*-per-hour case materialization with score/event-ID
  ordering and no access to evaluation truth.

The backend architecture and security boundaries are documented in
[docs/backend_architecture.md](docs/backend_architecture.md), with the frozen
results and limitations in the [Sprint 2 model card](docs/model_card.md).

The completed full-source run scored all 762,103 held-out test rows (1,561
positives): PR-AUC 0.1398, ROC-AUC 0.9695, and F1 0.1954 at the
validation-selected threshold. At 25 reviews per active hour it recovered
139/181 mapped laundering cases (76.80%) and 73/81 first-seen cases (90.12%).

## Sprint 3 analyst frontend

The frontend is one responsive operations workspace with a persistent sidebar
and role-aware routes. It keeps the analyst path clear: open the ranked queue,
claim a case, review only the evidence assigned by the study protocol, and
record a defensible disposition. Monitoring and governance views remain in the
same shell so model oversight does not interrupt the investigation workflow.

The detailed route map, evidence rules, conflict behavior, and authentication
design are documented in
[docs/frontend_architecture.md](docs/frontend_architecture.md).

## Native setup (no Docker)

Python 3.12 or newer is required.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

Acquire and verify the governed source data if it is not already installed:

```powershell
.\.venv\Scripts\python.exe scripts\acquire_data.py download ibm_aml_hi_small --accept-license
.\.venv\Scripts\python.exe scripts\acquire_data.py verify ibm_aml_hi_small
```

Train, evaluate, and package the full-source model:

```powershell
.\.venv\Scripts\python.exe scripts\train_sprint2.py --config config\model.json --bundle-id sprint2-lightgbm-v1
```

The command refuses the bounded `dev-100000` matrix, scores both chronological
holdouts in full, writes `artifacts/models/sprint2-lightgbm`, and verifies the
finished bundle. Use `--quick --output <new-directory>` for a toy end-to-end
dependency smoke test; quick-mode results are not scientific evidence.

Configure Neon:

```powershell
Copy-Item .env.example .env
# Edit .env: set DATABASE_URL and a unique JWT_SECRET_KEY.
.\.venv\Scripts\python.exe -m alembic upgrade head
$env:ADMIN_PASSWORD = '<use-a-secret-manager-or-temporary-shell-value>'
.\.venv\Scripts\python.exe scripts\seed_admin.py --email admin@example.com
.\.venv\Scripts\python.exe scripts\register_model.py --bundle artifacts\models\sprint2-lightgbm --created-by admin@example.com --activate
```

Use the pooled Neon URL for the application, for example
`postgresql://USER:PASSWORD@ep-name-pooler.REGION.aws.neon.tech/neondb?sslmode=require`.
An optional direct `DATABASE_MIGRATION_URL` may be used by Alembic. The runtime
normalizes URLs to SQLAlchemy's asyncpg dialect and enforces TLS for Neon.

Start the API:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Then open `/docs`, or check `/health/live` and `/health/ready`.

Start the analyst console in a second terminal (Node.js 20.19 or newer):

```powershell
Set-Location frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`. During development, Vite proxies `/api` and
`/health` to the API on `http://127.0.0.1:8000`; production deployments may set
`VITE_API_BASE_URL` as described in `frontend/.env.example`.

## API surface

All operational routes are under `/api/v1` and use short-lived bearer access
tokens. Refresh tokens rotate and reuse revokes the token family.

| Area | Routes |
| --- | --- |
| Authentication | login, refresh, logout, current user |
| Transactions | create, list, retrieve masked transactions |
| Predictions | single and bounded batch scoring with explanations |
| Cases | materialize hourly top-*N*, list/detail, claim, start review, disposition |
| Models | active-model metadata and admin-only promotion |
| Monitoring | current drift state, history, human-gated retraining requests |
| Audit | admin-only sanitized event history |

Roles are `fraud_analyst`, `senior_investigator`, and `admin`. Account values
are masked before persistence, request IDs are returned on responses, and
security-sensitive actions create append-only audit events.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\verify_sprint1.py --run-name dev-100000 --no-write
.\.venv\Scripts\python.exe -m alembic upgrade head

Set-Location frontend
npm run lint
npm run typecheck
npm test
npm run build
```

The test suite covers Sprint 1 invariants, model feature/split contracts,
artifact tamper detection, security tokens and refresh rotation, migrations,
SQLite ORM integration, OpenAPI, authenticated API workflows, role checks,
prediction persistence, and idempotent hourly queue materialization.

## Important outputs

| Path | Purpose |
| --- | --- |
| `config/model.json` | Frozen Sprint 2 model/training contract |
| `artifacts/models/sprint2-lightgbm/` | Checksummed deployable model bundle |
| `artifacts/models/sprint2-lightgbm/metrics.json` | Validation, test, calibration, and capacity metrics |
| `backend/app/main.py` | FastAPI application factory and runtime entry point |
| `backend/app/db/migrations/` | Neon-compatible Alembic schema |
| `frontend/src/App.tsx` | Authenticated, role-aware frontend route map |
| `frontend/src/features/cases/` | Ranked queue, evidence, and disposition workflow |
| `docs/backend_architecture.md` | Backend, security, and model lifecycle |
| `docs/frontend_architecture.md` | Frontend workflow, study arms, and API behavior |
| `docs/model_card.md` | Frozen model results, intended use, and limitations |
| `docs/sprint1_audit.md` | Sprint 1 completion decision and qualification |
| `reports/sprint2_backend_verification.json` | Final model/backend verification evidence |
| `reports/eda/summary.json` | Full-source EDA evidence |
| `reports/runs/dev-100000/verification.json` | Bounded Sprint 1 acceptance evidence |

## Research limitations

AMLworld is synthetic. Its 5,078,345 transactions, complete labels, and 370
laundering attempts make controlled evaluation possible, but its prevalence,
entities, behaviour, and time span are not production estimates. A strong
held-out result supports this implementation on AMLworld; it does not establish
real-world deployment effectiveness. Retraining and model promotion are
therefore human-gated rather than automatic.
