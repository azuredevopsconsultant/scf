# SCF Cohort Matrix — Databricks MLOps Pipeline

Production-grade MLOps platform for the SCF savings cashflow GLM model, built as a
Databricks Asset Bundle (DAB). Forecasts savings account balances up to 60 months
forward per cohort × product, achieving **2.28% Mean APE** on a £9–11bn portfolio.

**Model:** 30 GLMs (10 products × 3 GLMs) packaged as one MLflow pyfunc.

```
scf-cohort-dab/
├── databricks.yml                 # bundle root: dev/preprod/prod targets, service principals
├── .github/workflows/deploy.yml   # CI/CD: OIDC, unit tests → validate → deploy
├── resources/
│   ├── training_job.yml           # Training pipeline (quarterly, 14 tasks)
│   ├── inference_job.yml          # Inference pipeline (daily, 8 tasks + auto-retrain)
│   ├── maintenance_job.yml        # Delta OPTIMIZE (weekly)
│   └── integration_test_job.yml   # Integration tests (preprod)
│   └── inference_job.yml          # Job 2: inference pipeline
├── src/
│   ├── common/                    # shared, imported by both pipelines
│   │   ├── config.py              # UC table/volume name resolution
│   │   ├── glm_core.py            # ported DataPrep/GLMTrainer/.../ModelingPipeline
│   │   ├── pyfunc_model.py        # MLflow pyfunc wrapper (SCFCohortModel)
│   │   ├── mlflow_utils.py        # Champion/Challenger alias logic
│   │   ├── notifications.py       # conditional/rich email sends
│   │   ├── quality_checks.py      # reusable data-quality checks
│   │   └── drift_checks.py        # PSI-based drift detection
│   ├── training/
│   │   ├── 01_data_ingestion.py
│   │   ├── 02_data_preprocessing.py
│   │   ├── 03_model_training.py
│   │   ├── 04_model_evaluation.py
│   │   ├── 05_model_registration.py
│   │   └── 06_confirmation_mail.py
│   └── inference/
│       ├── 01_data_ingestion.py
│       ├── 02_data_preprocessing.py
│       ├── 03_inference.py
│       ├── 04_model_monitoring.py
│       ├── 05_data_drift.py
│       ├── 06_data_quality.py
│       └── 07_confirmation_mail.py
├── tests/
│   └── test_glm_core.py           # unit tests for the ported classes
├── requirements.txt               # runtime deps (installed on serverless jobs)
└── requirements-dev.txt           # -r requirements.txt + local/CI-only tooling
```

## Pipeline 1 — Training (Quarterly)

`feature_engineering → data_ingestion → data_preprocessing → data_validation → model_training (HPO+CV) → model_evaluation → model_validation → FMC gate → model_registration → deploy → smoke_test → auto_rollback → confirmation_mail`

- Runs quarterly (1st Jan/Apr/Jul/Oct 03:00 UTC).
- Hyperparameter grid (5 configs × 10 products) with 3-fold rolling-origin CV; each trial a nested MLflow child run.
- FMC second-line validation gate — human APPROVED/REJECTED via `fmc_approve.py` (24h polling).
- Every run registers `@challenger`; promoted to `@champion` only if MAPE improves >15%.

## Pipeline 2 — Inference
`data_ingestion → data_preprocessing → inferencing (applyInPandas) → model_monitoring → data_drift (PSI) → data_quality → retraining_trigger → auto_retrain → confirmation_mail`

- Runs daily (06:00 UTC).
- Always scores using the `@champion` alias — never `@challenger`.
- Distributed inference via Spark `applyInPandas` grouped by product (10 parallel executors).
- Output includes APE columns (`ape_balance`, `ape_receipts`, ...) and GLM ratio columns.
- 3-signal auto-retrain: PSI ≥ 0.25 OR null rate >10% OR MAPE >20% → fires training pipeline.

## CI/CD — GitHub Actions (OIDC, no PAT)

```
feature/* → PR      → unit-tests → validate → plan
develop   → push    → unit-tests → validate → deploy-dev
release/* → push    → deploy-preprod → integration-tests (5 stages)
main      → push    → deploy-prod (FMC sign-off gate)
```

GitHub authenticates to Databricks via OIDC token exchange — no PAT or secret in the repo.
Configure GitHub Environment variables `DATABRICKS_HOST` + `DATABRICKS_CLIENT_ID` per environment.

## Deploying

```bash
databricks bundle validate -t dev
databricks bundle deploy -t dev
databricks bundle run scf_training_pipeline -t dev
databricks bundle run scf_inference_pipeline -t dev

# once validated:
databricks bundle deploy -t prod
```

Before first deploy:
1. Create the secret scope used by `src/common/notifications.py`:
   ```bash
   databricks secrets create-scope scf-cohort-notifications
   databricks secrets put-secret scf-cohort-notifications smtp-user
   databricks secrets put-secret scf-cohort-notifications smtp-password
   ```
2. Update `databricks.yml` — `workspace.host`, `notification_email`, `catalog`/`schema`, and the prod `service_principal_name`.
3. Grant the job's run-as identity `USE CATALOG` / `USE SCHEMA` / `CREATE TABLE` / `CREATE VOLUME` on the target UC catalog, plus registry permissions on the model.

## Best practices applied here (and why)

**Bundle structure**
- One YAML per job under `resources/`, included via `include: resources/*.yml` in `databricks.yml`, rather than one giant file — keeps diffs small in code review and lets each pipeline evolve independently.
- `targets.dev` uses `mode: development` (isolated per-developer runs, job paused by default); `targets.prod` runs as a **service principal**, not a named user, so the job doesn't break when someone leaves the team.

**Notifications — three layers, each for a different purpose**
1. **Job-level `email_notifications`** (`on_success`/`on_failure`) — the reliable, zero-code "pipeline outcome" signal. Fires exactly once regardless of which task failed.
2. **Task-level `email_notifications`** on the highest-stakes task (`model_registration`) — extra visibility on the one step that changes what's in production.
3. **Custom conditional sends** (`src/common/notifications.py`) — for anything a fixed job-state event can't express: drift-only alerts, and a run summary that always fires via `run_if: ALL_DONE` even after an upstream failure.

Avoid building all alerting as custom code — native job notifications are simpler, need no SMTP credentials, and can't silently break if a helper module has a bug.

**Champion/Challenger governance**
- Uses Unity Catalog Model Registry **aliases**, not the deprecated stage-based registry.
- Every training run always registers + aliases `challenger` — there's a full audit trail of every candidate, even rejected ones.
- Promotion to `champion` is threshold-gated and logged with a reason string, not automatic on "any" success. Tighten this further (e.g. require manual approval) for regulated model-risk environments — swap `evaluate_promotion()`'s auto-promote for a task that just writes the recommendation and stops, with a human running `promote_challenger_to_champion()` explicitly.
- Inference always loads `@champion`, never `@challenger`, so a bad candidate can never affect production scoring.

**Data quality & drift**
- Quality checks (`src/common/quality_checks.py`) are dependency-light (no Great Expectations/Deequ requirement) — swap in your platform's standard framework if you already run one elsewhere.
- Drift uses PSI (population stability index), the standard metric for banking/credit-risk model monitoring, with a frozen **reference snapshot** captured once at training time — not "yesterday's batch" — so drift is measured against a stable baseline rather than a moving target.
- Both write to append-only Delta tables (`drift_results`, `data_quality_results`, `monitoring_metrics`) for audit and for a BI dashboard / Databricks SQL alert to sit on top of.

**Code organisation**
- `src/common/glm_core.py` is a direct, tested port of the notebook's classes — training and inference call the *same* functions for feature binning (`RATE_BINS`/`RATE_LABELS`) and projection math (`BuildProjections`), eliminating the two-copies-of-the-same-logic risk that existed across cells 37 and 46 of the original notebook.
- Config (table/volume names, thresholds) lives in one `config.py`, resolved from job parameters — promoting dev → prod is a variable override, not a code change.
- Every ingestion/preprocessing task ends with an `assert` on row counts — fail fast and loud rather than silently propagating an empty upstream extract.

**What to add before production sign-off**
- CI: run `pytest tests/` and `databricks bundle validate` on every PR (GitHub Actions / Azure DevOps), deploy to `dev` on merge to `main`, promote to `prod` via a tagged release.
- Unity Catalog Lakehouse Monitoring is a first-class alternative to the custom PSI check here if your workspace has it enabled — it adds automatic dashboards and profiling for free.
- Row-/column-level access controls on the PII base data (the original S3 path is named `...-pii-data`) — enforce via UC table grants, not notebook-level filtering.
