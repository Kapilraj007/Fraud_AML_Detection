# Research gap and publication claim

## What is not novel

A study framed only as “fraud score versus fraud score plus SHAP” would repeat established work. Jesús et al., in [*How can I choose an explainer? An Application-grounded Evaluation of Post-hoc Explanations*](https://arxiv.org/abs/2101.08758), evaluated data-only, model-score, and score-plus-explanation conditions with fraud analysts. Amarasinghe et al. later showed in [*On the Importance of Application-Grounded Experimental Design for Evaluating Explainable ML Methods*](https://arxiv.org/abs/2206.13503) that seemingly small design choices can change conclusions and found no evidence of incremental utility for the tested explanation methods in their deployment-grounded fraud task.

Those studies make two boundaries clear:

1. generic feature attribution is not itself a publishable contribution; and
2. offline explanation fidelity is not evidence that an explanation improves analyst decisions.

This project therefore does **not** claim that adding SHAP to a score is new, nor that an attractive explanation is automatically useful.

## Candidate contribution

The refined research question, recorded as the source of truth in [`config/study.json`](../config/study.json), is:

> Under temporal and AML-typology shift, does reliability-gated case-level context improve laundering-case recovery per analyst hour over score-only and generic feature-attribution review at a fixed queue capacity?

The candidate contribution combines five elements that must be evaluated together:

- **Case-level AML reasoning.** Review context links temporally related transfers and graph motifs instead of explaining each transaction in isolation.
- **Reliability gating.** The system may expose, qualify, or defer an explanation according to calibration, uncertainty, and shift evidence. It does not present every attribution as equally trustworthy.
- **Explicit shift evaluation.** Performance is stratified by chronological window and laundering typology, with transaction format and currency as secondary slices.
- **Finite analyst capacity.** The principal comparison is case recovery at the same hourly review budget, not unconstrained row-level AUROC.
- **Application-grounded outcomes.** Precision, false escalations, first-detection delay, backlog age, and recovered laundering value per analyst hour supplement predictive metrics.

The primary benchmark is [IBM AMLworld](https://github.com/IBM/AML-Data), introduced by Altman et al. in [*Realistic Synthetic Financial Transactions for Anti-Money Laundering Models*](https://papers.neurips.cc/paper_files/paper/2023/file/5f38404edff6f3f642d6fa5892479c42-Paper-Datasets_and_Benchmarks.pdf). Its transaction graph, complete synthetic labels, and laundering-pattern file permit controlled case and typology evaluation. They do not make it representative of a live institution.

## Falsifiable hypotheses

- **H1 — capacity:** reliability-gated case context improves `case_recall_at_fixed_capacity` over score-only and generic-explanation review on the chronologically latest holdout.
- **H2 — shift:** its advantage is retained, or degrades less, on typologies and temporal regimes that are underrepresented in training.
- **H3 — workload:** any gain is not bought by an unacceptable increase in false escalations, positive-backlog age, or review time.
- **H4 — gate value:** the reliability gate outperforms presenting the same type of context unconditionally, particularly on low-reliability slices.

H4 requires an additional ungated case-context ablation before a publication claim can be made. Sprint 1 establishes the data, leakage, split, queue, and evaluation contracts; it does not yet supply evidence for these hypotheses.

## Required publication evidence

A paper should not be drafted as a positive-result report until all of the following exist:

1. a preregistered or version-pinned analysis plan and frozen `config/study.json`;
2. a model and calibration procedure trained without access to the held-out period;
3. a declared reliability gate and an ungated-context ablation;
4. repeated or uncertainty-aware estimates on chronological and typology slices;
5. a power analysis and ethics review for any human-analyst study;
6. results reported for every configured arm, including null or adverse findings; and
7. replication on a legally usable second dataset or a clearly stated single-generator limitation.

PaySim may be used only as a fraud/domain-shift stress test; it is not an AML case benchmark. IEEE-CIS remains optional and restricted by its competition rules. Neither dataset should be pooled with AMLworld as though its labels represented the same outcome.

## Threats to validity

- **Synthetic realism:** AMLworld can encode generator-specific shortcuts, and its laundering prevalence is not a production base rate.
- **Case annotation:** pattern-file case IDs and typologies are privileged ground truth. They are reserved for stratification and evaluation, never prediction or queue ranking.
- **Analyst-time proxy:** Sprint 1 can compute outcomes per configured review-time assumption, but only a human study can estimate actual time and cognitive load.
- **Shift construction:** a chronological split is necessary but does not guarantee meaningful operational drift. Shift slices must be measured and reported.
- **Explanation dependence:** case context and reliability must be specified independently enough to permit ablation; otherwise their individual effects cannot be identified.

