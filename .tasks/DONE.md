# Done

## TASK-003: Implement Sprint 3 analyst frontend
**Priority:** P0 | **Tags:** frontend, react, ux, monitoring, accessibility

Build the Sprint 3 analyst-facing frontend as a clean, professional single-dashboard experience with sidebar navigation. Implement authentication, a ranked auto-refreshing case queue, case review and disposition workflow, stable A/B explanation treatment, accessible risk/explanation visuals, drift monitoring, and human-gated retraining controls integrated with the Sprint 2 FastAPI backend.

### Done

- Built a typed, responsive React analyst console with a corporate command center, persistent sidebar, role-aware routes, rotating-token authentication, and resilient loading, error, empty, and stale-state handling.
- Implemented the ranked queue-to-disposition workflow with masked transaction context, fixed study-arm evidence, accessible signed feature contributions, reliability gating, conflict recovery, and materialization-time prediction pinning.
- Added active-model-scoped KS/ADWIN monitoring, human-gated retraining requests, transaction search, and admin-only model and sanitized audit views.
- Verified ESLint, TypeScript, 13 frontend tests, a code-split production build, eight desktop/mobile browser smoke routes, the Alembic upgrade/downgrade path, and 76 backend tests plus 5 subtests.

---

## TASK-002: Complete Sprint 2 model and backend
**Priority:** P0 | **Tags:** backend, ml, api, neon, security, monitoring

Implement the complete Sprint 2 backend for the capacity-aware AML triage system: select and train the fraud model, validate it on chronological holdouts, build explainability and drift services, expose the versioned FastAPI API, persist operational records in Neon PostgreSQL, implement JWT role-based access and audit trails, and provide migrations, tests, and run documentation without Docker.

### Done

- Audited Sprint 1, preserved its passing implementation gate, and documented why the zero-positive `dev-100000` test split cannot support model evaluation.
- Trained and checksum-versioned calibrated LightGBM, logistic, and Isolation Forest artifacts on the full 5,078,345-row governed source; full validation/test and capacity/case metrics are recorded in the model card.
- Implemented the FastAPI scoring, explanation, transaction, hourly queue, case, monitoring, model, authentication/RBAC, refresh rotation, and audit workflows.
- Added Neon-pooled async persistence, the complete Alembic schema, admin/model deployment CLIs, drift/retraining services, and native no-Docker configuration.
- Verified 76 tests plus 5 subtests, Sprint 1 read-only acceptance, bundle loading/scoring, clean migration/seed/registration, and live Uvicorn readiness.

---

## TASK-001: Implement Sprint 1 foundation and data pipeline
**Priority:** P0 | **Tags:** data, research, reproducibility

Implement the complete Sprint 1 scope from the supplied SKI027 plan: repository foundation, governed dataset acquisition, leakage-safe feature engineering, chronological feature store and splits, EDA/data dictionary, and a capacity-constrained case-queue harness aligned to a defensible research gap.

### Done

- Acquired and verified all three IBM AMLworld HI-Small artifacts with governed provenance and license records.
- Implemented globally ordered, point-in-time features, pattern joins, chronological splits, train-only preprocessing, and a ground-truth-isolated hourly case queue.
- Produced full-source EDA, a measured data dictionary, research-gap/methodology documentation, and a reproducible notebook.
- Added create-only orchestration and an independent acceptance verifier; the materialized `dev-100000` run passes all checks.
- Verified the implementation with 55 tests plus 5 subtests; Docker context excludes governed data, while an image build remains unexecuted because the local daemon is unavailable.

---
