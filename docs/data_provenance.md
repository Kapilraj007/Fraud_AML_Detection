# Data provenance and acquisition

Sprint 1 uses a governed registry rather than embedding raw data in the repository. The primary benchmark is IBM AMLworld HI-Small. PaySim is an optional fraud-only stress test, and IEEE-CIS is an optional restricted benchmark. A successful download produces `data/raw/<dataset-id>/manifest.json`; raw files and manifests should be retained as local research evidence and must not be committed unless their redistribution terms have been reviewed.

## Reproducible commands

The acquisition utility requires only the Python standard library.

```text
python scripts/acquire_data.py list
python scripts/acquire_data.py download ibm_aml_hi_small --accept-license
python scripts/acquire_data.py verify ibm_aml_hi_small
```

To limit the primary download to one artifact, repeat `--file` as needed:

```text
python scripts/acquire_data.py download ibm_aml_hi_small --accept-license --file HI-Small_Trans.csv
```

Files stream to a sibling `.part` file. A file is atomically moved into place only after extraction and SHA-256 verification. Kaggle may wrap a large single-file response in a ZIP; the tool validates every archive path and rejects absolute paths, parent traversal, links, and path collisions before extracting the expected member. `--force` is required to replace an existing file.

For authenticated Kaggle access, prefer `KAGGLE_API_TOKEN` or `~/.kaggle/access_token`. Legacy `KAGGLE_USERNAME` plus `KAGGLE_KEY`, and the equivalent fields in `~/.kaggle/kaggle.json`, remain supported. `KAGGLE_CONFIG_DIR` is honored. Credentials are never written to the manifest, logged, sent to a non-Kaggle origin, or forwarded across an origin-changing redirect.

## Primary: IBM AMLworld HI-Small

- Authoritative project: <https://github.com/IBM/AML-Data>
- Kaggle distribution: <https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml>
- Dataset license: [CDLA-Sharing-1.0](https://cdla.dev/sharing-1-0/). The Apache-2.0 badge on IBM's GitHub repository covers repository code, not the data.
- Research reference: Erik Altman, Jovan Blanuša, Luc von Niederhäusern, Béni Egressy, Andreea Anghel, and Kubilay Atasu, *Realistic Synthetic Financial Transactions for Anti-Money Laundering Models* (NeurIPS 2023 Datasets and Benchmarks).

The registry requests the three HI-Small artifacts individually from Kaggle instead of downloading the roughly 40 GB multi-variant archive:

| Artifact | Purpose | Governed SHA-256 |
| --- | --- | --- |
| `HI-Small_Trans.csv` | Transaction graph and laundering labels | `b19d39f515523373f991b689c07e11e7b0b95c17a2c27a87d91584ae16c5b040` |
| `HI-Small_Patterns.txt` | Typology/ring ground truth | `2c546b5ce6009e73851f0139af053cf845f08bf92f3bc82fe1eb937dec2ef39b` (current Kaggle bytes); removing the single trailing LF gives the earlier published digest `7636c1d712168139ba0ff90f1b45aac9888d0ad46387084560886f204c03d6e6` |
| `HI-Small_accounts.csv` | Account-to-entity reference | Upstream does not publish a trusted digest; this run pins 34,053,187 bytes at `786808526e33cfc441212dd6fccda7edfc24172149bed59c6ef59b186836b014` in the local manifest |

The source endpoints are the official Kaggle dataset-file API form:

```text
https://www.kaggle.com/api/v1/datasets/download/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/<filename>
```

HI-Small is synthetic, not anonymized bank evidence. Its complete labels make controlled evaluation possible, but the generator's behavior, institution mix, time span, and laundering prevalence must not be presented as production prevalence or proof of performance on real customers.

## Optional: PaySim

- Simulator repository: <https://github.com/EdgarLopezPhD/PaySim>
- Kaggle distribution and metadata: <https://www.kaggle.com/datasets/ealaxi/paysim1>
- License: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- Endpoint: `https://www.kaggle.com/api/v1/datasets/download/ealaxi/paysim1/PS_20174392719_1491204439457_log.csv`

PaySim simulates mobile-money **fraud**, not money laundering. It may be used only as a secondary robustness/domain-shift experiment and must not be pooled with IBM labels as if the targets meant the same thing. For predictive features, exclude `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`, and the rule-generated `isFlaggedFraud`; these expose simulator bookkeeping or post-outcome information. `isFraud` is the evaluation target and must never be an input feature. Fit encoders and thresholds on training data only.

## Restricted optional benchmark: IEEE-CIS

- Data page: <https://www.kaggle.com/competitions/ieee-fraud-detection/data>
- Binding rules: <https://www.kaggle.com/competitions/ieee-fraud-detection/rules>
- License shown by Kaggle: **Subject to Competition Rules**

Each user must sign in, join the competition, and accept its rules in the Kaggle UI before using the API. The CLI checks for credentials and uses only Kaggle's authenticated competition-file endpoint. `--accept-license` records a local acknowledgement; it does not accept the competition rules remotely and cannot bypass Kaggle's access control. A `401` or `403` is treated as an access failure, not something to work around. Competition data is limited to **non-commercial** competition participation, academic research, and education under rule 7A. Do not commit, publish, transmit, or redistribute it to anyone who has not independently accepted the rules. On POSIX, the downloader creates the dataset directory and files with owner-only permissions (`0700`/`0600`); on Windows, the operator must verify the directory ACL. The rules require suitable access controls; consult the current rules before any publication.

## Manifest contract

Each installed artifact has a manifest record containing:

- dataset ID, human-readable name, and dataset version;
- provider, landing page, license identifier, and license URL;
- exact source URL and UTC `fetched_at` time;
- byte count, observed SHA-256, and any pre-registered expected SHA-256;
- an explicit `license_acknowledged` marker.

Run `verify` before every pipeline run and before publishing results. Registry hashes take precedence. Files without an upstream trusted digest use trust-on-first-download: the tool hashes the first non-empty payload and pins that observation in the acquisition manifest. Later downloads must match it. This detects later change but does not independently authenticate the first payload. If a source changes, the downloader leaves the installed file and manifest untouched; do not silently rotate a digest. Quarantine the new artifact, compare its schema/content and upstream version notes, document the decision, and then review a registry change.

## Access constraints at implementation time

All three public IBM HI-Small artifacts were acquired in this workspace and `verify ibm_aml_hi_small` passes against the local manifest. No Kaggle credentials or manual competition-rule acceptance are bundled with the repository, and future public downloads may still be authenticated or rate-limited by Kaggle. Configure personal credentials if that occurs; no mirror is silently substituted. The IBM account artifact lacks an independently published upstream SHA-256, so its observed digest is trust-on-first-download and remains manifest-scoped until a second authoritative source is reconciled.

Restricted IEEE-CIS bytes were not acquired. Each user must accept that competition's rules personally; the implementation cannot do so on another user's behalf. Offline acquisition behavior is covered with local fixtures without redistributing licensed data.
