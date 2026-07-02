# SCF Cohort Matrix — Databricks Asset Bundle

Two Databricks Jobs (multi-task workflows) generated from the original
`scf_cohort_matrix.ipynb` GLM cohort-modelling notebook.

```
scf-cohort-dab/
├── databricks.yml                 # bundle root: targets (dev/prod), variables
├── resources/
│   ├── training_job.yml           # Job 1: training pipeline
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
└── requirements-dev.txt
```

## Pipeline 1 — Training

`Data Ingestion → Data Preprocessing → Model Training → Model Evaluation → Model Registration (Champion/Challenger) → Confirmation Mail`

- Runs weekly (Mon 03:00 UTC) — adjust `resources/training_job.yml` schedule to your data refresh cadence.
- Job-level `email_notifications` fire on **on_success** and **on_failure**, plus a duration-warning alert.
- Every successful run registers a new Unity Catalog model version aliased `challenger`. It's only promoted to `champion` if it beats the current champion's portfolio MAPE by `MAPE_PROMOTION_THRESHOLD` (default 15%, see `src/common/config.py`).

## Pipeline 2 — Inference

`Data Ingestion → Data Preprocessing → Inferencing → Model Monitoring → Data Drift (alert+mail) → Data Quality → Confirmation Mail`

- Runs daily (06:00 UTC).
- Always scores using the `champion` alias — never `challenger` — so production traffic is isolated from unpromoted candidates.
- Data-drift task sends an email **only when PSI crosses the alert threshold** (0.25 by default), separate from the always-on confirmation mail.
- Confirmation mail uses `run_if: ALL_DONE` so it fires and reports pass/fail per stage even if an upstream task failed.

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
