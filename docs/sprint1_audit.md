# Sprint 1 completion audit

Audit date: 2026-10-13

## Decision

Sprint 1 is complete as an implementation and bounded acceptance milestone for
the repository's revised IBM AMLworld design. It is not a completed full-scale
feature-matrix experiment, and the bounded `dev-100000` matrices must not be
used to claim model performance.

## Evidence that passes

- All governed IBM AMLworld HI-Small inputs are installed and their recorded
  SHA-256 digests verify.
- Full-source EDA covers 5,078,345 transactions, including 5,177 positive
  labels, 370 laundering attempts, and eight pattern typologies.
- The create-only `dev-100000` run materializes the feature store,
  chronological splits, train-fitted preprocessing, model matrices, analyst
  queue, and sealed evaluation key.
- `reports/runs/dev-100000/verification.json` reports `status: pass`; it also
  confirms the reviewer artifact is separated from ground truth and respects
  the fixed hourly capacity.
- Sprint 1's pre-backend automated suite passed 55 tests, and the independent
  verifier passed in read-only mode.

## Qualification

The implementation plan originally named IEEE-CIS and PaySim. The repository
instead adopted IBM AMLworld as the primary governed source because it includes
transaction labels and laundering-pattern case membership; that change is
documented in the study and provenance contracts.

Only the 100,000-row bounded run has a complete materialized temporal feature
store and matrix in this workspace. Its chronological model splits contain:

| Split | Rows | Positives |
| --- | ---: | ---: |
| Train | 65,693 | 5 |
| Validation | 10,946 | 2 |
| Test | 23,361 | 0 |

The zero-positive test period makes this development artifact unsuitable for
evaluation. Sprint 2 therefore trains a compact, label-blind raw-feature model
against the full 5,078,345-row source, keeps negative undersampling confined to
the training period, and scores the complete chronological validation and test
periods.

## Gate to Sprint 2

Proceeding to Sprint 2 is justified because the acquisition, provenance,
leakage controls, timestamp-group split semantics, queue policy, and evaluation
isolation are implemented and tested. Scientific claims remain gated on the
full-source Sprint 2 holdouts and the versioned model report.
