# Work Log

Completed work is recorded here with the newest entry first.

## 2026-10-14 — TASK-003

Completed the Sprint 3 analyst frontend as one responsive, role-aware corporate workspace spanning the command center, ranked investigation workflow, study-controlled evidence, drift monitoring, retraining requests, transactions, and governance. Evidence: ESLint and TypeScript pass; 13 frontend tests pass; the code-split production build succeeds; eight desktop/mobile browser smoke routes pass without console errors, horizontal overflow, duplicate main landmarks, unlabeled form controls, or an exposed closed mobile sidebar; the complete backend remains green at 76 tests plus 5 subtests, including the new exact prediction-link migration and case-detail contract.

## 2026-10-13 — TASK-002

Completed the Sprint 2 model and operational backend. Evidence: the full-source `sprint2-lightgbm-v1` bundle passes checksum and runtime scoring verification; the 762,103-row held-out test reports PR-AUC 0.1398, ROC-AUC 0.9695, F1 0.1954, and 76.80% mapped case recall at 25 reviews per active hour; 76 tests plus 5 subtests pass; clean migration, admin seed, model activation, liveness, and readiness smokes pass. Neon is the configured production database and no Docker files are present; applying the migration to a live Neon project requires deployment credentials.

## 2026-10-09 — TASK-001

Completed the Sprint 1 AML foundation: governed full data acquisition, causal feature pipeline, chronological splits, train-only preprocessing, capacity-constrained queue, complete EDA/dictionary, publishable-gap study contract, orchestration, and independent artifact verification. Evidence: all source hashes pass, the `dev-100000` acceptance report is green, and 55 tests plus 5 subtests pass.
