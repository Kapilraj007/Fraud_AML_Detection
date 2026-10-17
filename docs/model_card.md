# Sprint 2 AML model card

Bundle: `sprint2-lightgbm-v1`  
Training date: 2026-10-13  
Primary model: LightGBM 4.7.0, best iteration 45  
Feature contract: `raw-baseline-v1` (23 label-blind features)

## Intended use

The model prioritizes synthetic AMLworld transactions for a fixed-capacity
analyst queue. It is a research and backend-integration baseline, not an
autonomous enforcement system and not evidence of performance on real customer
activity. Analysts make final decisions; drift-triggered retraining and model
promotion require human approval.

## Model choice

LightGBM is the selected primary model because it is efficient for imbalanced
tabular data, handles nonlinear interactions, supports deterministic CPU
training, provides native TreeSHAP contributions, and has a native non-pickle
serialization format. A portable L2 logistic-regression baseline and a
secondary Isolation Forest anomaly detector are included in the bundle.
XGBoost remains an optional robustness cross-check and was not enabled for this
frozen run.

## Data and split contract

IBM AMLworld HI-Small contains 5,078,345 synthetic transactions. Splits are
chronological and never divide an equal-timestamp group.

| Split | Period boundary | Rows | Positives |
| --- | --- | ---: | ---: |
| Train | before 2022-09-07 14:55 UTC | 3,554,649 | 2,856 |
| Validation | to 2022-09-09 03:16 UTC | 761,593 | 760 |
| Test | from 2022-09-09 03:16 UTC | 762,103 | 1,561 |

All training positives were retained. Training negatives alone were reduced to
285,600 with a deterministic 100:1 negative-to-positive ratio and
inverse-inclusion weights. Validation and test were scored in full. The Platt
calibrator and decision threshold were fitted on validation only; test labels
were read only for the final frozen evaluation.

## Frozen performance

The validation-selected F1 threshold is `0.0949273117`.

| Metric | Validation | Held-out test |
| --- | ---: | ---: |
| PR-AUC / average precision | 0.0901 | 0.1398 |
| ROC-AUC | 0.9541 | 0.9695 |
| F1 at frozen threshold | 0.1512 | 0.1954 |
| Precision at threshold | 0.1688 | 0.2102 |
| Recall at threshold | 0.1368 | 0.1826 |
| Brier score | 0.000948 | 0.001888 |
| Expected calibration error | 0.000069 | 0.000518 |

At a fixed capacity of 25 reviews per active hour, the held-out test produced:

- 2,204 reviews across 190 active hours;
- 764 positive alerts, 1,440 false escalations, 34.66% alert precision, and
  48.94% transaction-level positive recall;
- 139 of 181 mapped laundering cases detected (76.80% case recall); and
- 73 of 81 first-seen test cases detected (90.12% first-seen case recall).

Case metrics use the evaluation-only laundering-pattern membership artifact
after ranking. Of 1,561 positive test rows, 1,219 matched that artifact; the
remaining positive rows still contribute to transaction metrics but cannot be
assigned a pattern case. Reported recovered value is synthetic nominal value
and must not be interpreted as real financial impact.

## Features and leakage controls

Inputs cover paid/received amounts and transforms, amount consistency, same
bank/account and currency-mismatch flags, cyclic hour/weekday context, weekend
status, and stable 4,096-bucket hashes for banks, accounts, currencies, and
payment format. Labels, case IDs, typologies, pattern membership, analyst
outcomes, and post-event fields are prohibited from the feature contract and
online request schemas.

The compact raw-feature model is intentionally stateless so the full source can
be trained on bounded hardware and the exact transformation can be reproduced
online. It does not claim the richer historical graph/velocity feature set as a
full-source result.

## Explainability and reliability

Per-prediction feature contributions come from LightGBM's native TreeSHAP
output. Narratives are withheld or marked low-reliability when contributions
are negligible, drift is active, or supervised and anomaly signals disagree
strongly. Explanations describe model behaviour; they do not establish causal
reasons for laundering.

## Artifact and reproducibility

`artifacts/models/sprint2-lightgbm/manifest.json` records SHA-256, size, and
serialization for every payload. The primary model is native LightGBM text;
calibration, schema, configuration, metrics, training provenance, and the
logistic baseline are JSON. Isolation Forest is a checksum-verified trusted
local joblib artifact and requires an explicit loader opt-in.

Reproduce the frozen run with:

```powershell
.\.venv\Scripts\python.exe scripts\train_sprint2.py --config config\model.json --bundle-id sprint2-lightgbm-v1
```

The output directory is create-only, so archive or choose a new output path
before retraining.

## Limitations and monitoring

- AMLworld is synthetic and its prevalence and behaviour are not production
  estimates.
- Stable hashing can collide and provides no semantic representation of an
  entity.
- The frozen threshold reflects validation F1; a real deployment must set
  capacity and cost policy with compliance owners.
- Performance may vary under institution, geography, currency, typology, or
  temporal shift.
- KS feature checks and ADWIN score monitoring are available, but drift opens
  a human-reviewed retraining request and never deploys a model automatically.
