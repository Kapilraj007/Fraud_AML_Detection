# Sprint 1 methodology

## Scope

Sprint 1 creates a reproducible, leakage-resistant substrate for a capacity-aware AML triage study. It covers governed acquisition, integrity checks, streaming EDA, pattern-ground-truth parsing, causal feature construction, deterministic chronological splits, train-only preprocessing, and a label-safe top-*N*-per-hour review queue. Model training, calibrated reliability scores, generated explanations, and human-subject conclusions are outside this sprint.

The experimental contract is versioned in [`config/study.json`](../config/study.json). When prose and configuration differ, the versioned configuration and the manifest produced by the executed pipeline define the run.

## Data and provenance

The primary dataset is IBM AMLworld HI-Small. The registry in [`config/datasets.json`](../config/datasets.json) records its provider, license, source URLs, expected digests, and optional artifacts. The acquisition utility streams downloads to a temporary file, rejects unsafe archive paths, verifies SHA-256 before installation, and writes a local manifest.

The installed Sprint 1 sources are:

- `HI-Small_Trans.csv`, SHA-256 `b19d39f515523373f991b689c07e11e7b0b95c17a2c27a87d91584ae16c5b040`;
- `HI-Small_Patterns.txt`, SHA-256 `2c546b5ce6009e73851f0139af053cf845f08bf92f3bc82fe1eb937dec2ef39b`; and
- `HI-Small_accounts.csv`, 34,053,187 bytes with manifest-pinned SHA-256 `786808526e33cfc441212dd6fccda7edfc24172149bed59c6ef59b186836b014` (trust on first download because the upstream registry supplies no digest).

Run `python scripts/acquire_data.py verify ibm_aml_hi_small` before every build. See [`docs/data_provenance.md`](data_provenance.md) for license and source limitations.

## Streaming exploratory analysis

[`scripts/run_eda.py`](../scripts/run_eda.py) reads CSV chunks or Arrow record batches from a Parquet file/directory. It never concatenates the transaction table. The run writes:

- `summary.json`: scope, resolved semantic fields, time span, label balance, amount quantiles from a deterministic priority sample, entity cardinalities, provenance, and guardrails;
- `column_profile.csv` and `numeric_profile.csv`: measured types, missingness, and streaming moments;
- `categorical_top_values.csv`: the top values for bounded-cardinality operational fields;
- `daily_activity.csv`: daily volume and value;
- `typology_profile.csv`: pattern-file case and transaction counts by typology;
- `quality_checks.csv`: parse, sign, label, entity, and input-order checks; and
- `class_balance.png`, `daily_volume.png`, and `amount_distribution.png`.

Entity cardinalities use deterministic 64-bit hashes to bound retained object memory and can theoretically collide. Amount quantiles are approximate because they come from a fixed-size uniform priority sample; row/class counts, null rates, streaming moments, and daily aggregates are exact for the rows in scope. A run with `--max-rows` is explicitly marked as a development sample.

The raw AMLworld file is not assumed to be chronological. The EDA counts adjacent timestamp regressions, and the feature pipeline sorts events before constructing history. A nonzero raw-order warning is therefore an expected reason to sort, not permission to compute causal windows on file order.

## Pattern annotations and leakage boundary

The pattern file groups labeled transactions into synthetic laundering attempts and names their typologies. Its transaction fields are matched to the raw transactions using deterministic fingerprints. The feature store represents possible multi-case membership as `laundering_case_ids`, `laundering_typologies`, and `laundering_case_positions`; all are evaluation metadata only.

These fields, together with `label`, `Is Laundering`, investigator outcomes, and any post-event field, are prohibited model inputs. Reviewer-facing queue files are physically separated from the evaluation key containing ground truth. The generated data dictionary marks such fields with `leakage_risk_flag=true`.

## Causal feature contract

All rolling and graph features are created by the shared [`transform_features`](../backend/app/ml/transform_features.py) implementation. The invariant is the half-open interval:

```text
[event_time - window, event_time)
```

The current row and all other rows with the same timestamp are excluded from its history. Reusing the explicit feature state across non-decreasing chunks must give the same result as one chronological call. The feature families include:

- source counts and values over 1 hour, 24 hours, and 7 days;
- preceding-30-day source mean, amount ratio, and counterparty novelty;
- fan-in, fan-out, recent inbound activity, rapid pass-through, and causal chain depth; and
- adapter-specific PaySim behavior features when that optional dataset is selected.

Labels never participate in a feature calculation. PaySim balance fields and `isFlaggedFraud` are deliberately excluded. IEEE-CIS time remains relative and its composite entity key is explicitly a pseudo-entity; no calendar origin or destination graph is invented.

## Feature store and dictionary

Canonical rows are written as Parquet under `data/processed/feature_store`, partitioned by `event_date`. The build manifest records inputs, digests, configuration, row counts, schema, time range, and output files. Dataframe runtime state is not serialized into Parquet metadata.

[`scripts/build_data_dictionary.py`](../scripts/build_data_dictionary.py) profiles the processed feature store when present and otherwise profiles the raw CSV while appending the expected canonical contract. Its output includes, at minimum:

```text
feature_name,dtype,source_table,null_rate,transformation_applied,leakage_risk_flag
```

Additional role and description fields make the leakage policy reviewable. Empty null rates indicate expected fields that were not observed in the profiled input, rather than an asserted zero-null rate.

## Chronological split and preprocessing

Events are stably ordered by event time and physical `source_row_number`, then assigned contiguously using the ratios in [`config/pipeline.json`](../config/pipeline.json): 70% train, 15% validation, and 15% test. Boundaries and row counts are persisted. The split contract keeps equal-time rows together, rejects overlap and temporal inversion, and does not randomly move events from the future into training.

All learned preprocessing state—numeric medians, scales, category frequencies, and category vocabularies—is fitted on train only. Validation and test are transformed with the frozen training artifact. Unknown categories receive an explicit representation. Metadata, labels, case fields, and split keys remain outside the predictor matrix.

## Capacity-constrained queue

The queue policy is read from [`config/study.json`](../config/study.json). Within each event hour, candidates are ranked by a supplied label-blind score and broken deterministically by ascending `event_id`; only the configured capacity is selected. Sprint 1 may use the fixed label-blind development rule for harness validation, but that rule is not a trained-model result.

Study assignment is deterministic and case-consistent, falling back to event ID for transactions without a labeled case. Configured conditions are data-only, score-only, generic explanation, and reliability-gated case context. Sprint 1 creates these assignment fields but does not simulate human responses.

The primary endpoint is case recall at fixed capacity. Secondary endpoints include alert precision/recall, false escalations, time to first detection, positive-backlog age, and recovered laundering value per analyst hour. Metrics are calculated only through the sealed evaluation key.

## Reproducibility sequence

From the repository root with the pinned environment installed, the preferred
create-only command is:

```powershell
python scripts/run_sprint1.py
```

It verifies acquisition, creates every downstream artifact, runs the
independent audit, and refuses to overwrite an existing output. A bounded
wiring run is automatically isolated so it cannot be mistaken for the
canonical publication path:

```powershell
python scripts/run_sprint1.py --max-rows 100000
python scripts/verify_sprint1.py --run-name dev-100000
```

The equivalent individual stages are:

```powershell
python scripts/acquire_data.py verify ibm_aml_hi_small
python -m pipelines.build_feature_store
python -m pipelines.split_data
python -m pipelines.preprocess_splits
python scripts/run_eda.py --input data/processed/feature_store --output-dir reports/eda
python scripts/build_data_dictionary.py --input data/processed/feature_store --output docs/data_dictionary.csv
python -m pytest
```

For a fast wiring check, add `--max-rows` to the EDA and dictionary commands. Do not use a bounded-sample report as the final descriptive result.

## Sprint 1 acceptance gates

1. Raw installed bytes match the governed manifest.
2. EDA covers the complete source and records whether the input was sampled.
3. Pattern annotations parse without orphan or malformed blocks.
4. Equal-time rows cannot see one another in feature history.
5. Feature partitions are chronological and their manifest reconciles to output rows.
6. Train, validation, and test are disjoint and temporally ordered.
7. Preprocessing fitted on train yields identical validation/test transforms regardless of their labels.
8. The queue enforces hourly capacity and contains no target, case, typology,
   or pattern-ground-truth field; those exist only in the sealed evaluation key.
9. Tests pass from a clean environment using the pinned dependencies.
