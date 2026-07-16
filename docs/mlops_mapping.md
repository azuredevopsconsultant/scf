# Databricks Enterprise MLOps Mapping — SCF Cohort GLM

Complete mapping of all 47 Python files to MLOps pipeline stages.
Last updated: 2026-07-16

---

## E — Training Pipeline

| Step | File | What it does |
|---|---|---|
| Feature Engineering (Redshift) | `src/training/00_feature_engineering_redshift.py` | Spark JDBC read from Redshift, computes 7 sales aggregates, writes to feature_store schema |
| Data Ingestion | `src/training/01_data_ingestion.py` | Reads parquet/csv from UC Volume → Bronze Delta tables, pins Delta version |
| Data Preprocessing | `src/training/02_data_preprocessing.py` | Cohort aggregation, ratio features, dbb_range binning, FeatureEngineeringClient write |
| Data Validation | `src/training/02b_data_validation.py` | Row count, nulls, value ranges, schema, product coverage — fails on breach |
| Model Training | `src/training/03_model_training.py` | 5-config HPO grid, 3-fold rolling-origin CV, nested MLflow child runs, baseline benchmark |
| Model Evaluation | `src/training/04_model_evaluation.py` | MAPE/RMSE per product + baseline comparison, writes model_eval_metrics |
| Model Validation | `src/training/04b_model_validation.py` | 4 automated checks + business metrics chart (Challenger vs Champion) |
| FMC Validation Gate | `src/training/05a_fmc_validation_gate.py` | Human-in-the-loop: polls model_approvals table for 24h, Teams notification |
| FMC Approve (reviewer) | `src/training/fmc_approve.py` | FMC reviewer runs this to record APPROVED/REJECTED decision |
| Model Registration | `src/training/05_model_registration.py` | Archives rejected, registers pyfunc, @baseline/@challenger/@champion aliases, model_cards |
| Confirmation Mail | `src/training/06_confirmation_mail.py` | Email + Teams summary (always runs) |
| Model Deployment | `src/training/07_deploy_serving_endpoint.py` | 90%/10% gradual rollout, zero-downtime, writes deployment_history |
| Smoke Test | `src/training/08_smoke_test_serving_endpoint.py` | 1 real row → endpoint, asserts non-empty response |
| Auto Rollback | `src/training/09_auto_rollback_guard.py` | Reverts @champion alias + endpoint if smoke test fails, writes rollback_events |

## G — Inference Pipeline

| Step | File | What it does |
|---|---|---|
| Data Ingestion | `src/inference/01_data_ingestion.py` | Reads from UC Volume (parquet), overwrites Bronze tables |
| Data Preprocessing | `src/inference/02_data_preprocessing.py` | Same feature transforms as training, seeds balance_lag_1, FeatureEngineeringClient merge |
| Batch Inference | `src/inference/03_inference.py` | applyInPandas per product, APE columns, ratio columns, MLflow lineage run |
| Model Monitoring | `src/inference/04_model_monitoring.py` | Actuals feedback loop, live MAPE/RMSE, writes monitoring_metrics |
| Drift Detection | `src/inference/05_data_drift.py` | PSI on 5 features vs frozen reference, WARN/ALERT, email+Teams on ALERT |
| Data Quality | `src/inference/06_data_quality.py` | Row count, nulls, schema, value ranges — fails on breach |
| Confirmation Mail | `src/inference/07_confirmation_mail.py` | Daily summary email (always runs) |
| Retraining Trigger | `src/inference/08_retraining_trigger.py` | 3-signal: PSI≥0.25 OR null>10% OR MAPE>20 → fires training pipeline |

## F — Unity Catalog (Governance Layer)

### Common Modules
| Module | File | Purpose |
|---|---|---|
| Configuration | `src/common/config.py` | 19 UC table paths (catalog.schema.table), feature_schema, model_schema, thresholds |
| GLM Core | `src/common/glm_core.py` | DataPrep, GLMTrainer (3 GLMs), PredictionBuilder, BuildProjections, Evaluator, RATE_BINS |
| MLflow Utils | `src/common/mlflow_utils.py` | register_challenger, promote_challenger_to_champion, archive_rejected, set_rich_version_tags, search_best_run |
| Pyfunc Model | `src/common/pyfunc_model.py` | SCFCohortModel wrapping 10 product lookup tables for MLflow pyfunc |
| Drift Checks | `src/common/drift_checks.py` | PSI computation (banking-standard), compute_drift_report, any_alerts |
| Quality Checks | `src/common/quality_checks.py` | check_row_count, check_null_rates, check_value_range, check_schema_columns |
| Retraining | `src/common/retraining.py` | evaluate_retraining_need (3-signal logic) |
| Notifications | `src/common/notifications.py` | Email + Teams webhook, conditional send, per-environment secret scope |
| Dataset Versioning | `src/common/dataset_versioning.py` | Delta version pin → dataset_versions table for VERSION AS OF replay |

### One-Time Setup
| File | Purpose |
|---|---|
| `src/dev/create_all_tables.py` | Creates all 16 UC Delta tables with typed schemas |
| `src/dev/setup_rbac.py` | GRANT permissions per role (ml-engineers, data-analysts, fmc-reviewers) |
| `src/dev/setup_lakehouse_monitoring.py` | Enables Lakehouse Monitoring on 4 tables |
| `src/dev/delta_maintenance.py` | OPTIMIZE + ZORDER + ANALYZE on all key tables |
| `src/dev/create_model_serving_endpoint.py` | Manual endpoint creation helper |
| `src/dev/observability_dashboard_queries.sql` | SQL queries for BI dashboards |

### Infrastructure as Code
| File | Purpose |
|---|---|
| `databricks.yml` | Bundle definition: dev/preprod/prod targets, service principals, secret scopes |
| `resources/training_job.yml` | Training pipeline DAG (quarterly, 14 tasks) |
| `resources/inference_job.yml` | Inference pipeline DAG (daily, 8 tasks + condition gate + auto_retrain) |
| `resources/maintenance_job.yml` | Delta maintenance job (weekly Sunday 02:00 UTC) |
| `resources/integration_test_job.yml` | Integration test job (runs in preprod before prod deploy) |

### CI/CD
| File | Purpose |
|---|---|
| `.github/workflows/deploy.yml` | GitHub Actions OIDC: unit-tests → validate → deploy-dev/preprod/prod |
| `scripts/setup_secret_scopes.ps1` | Windows: creates Databricks secret scopes (SMTP + Redshift + webhook) |
| `scripts/setup_secret_scopes.sh` | Linux/Mac equivalent |

### Tests
| File | Tests |
|---|---|
| `tests/test_glm_core.py` | DataPrep splits, Evaluator MAPE/RMSE, RATE_BINS |
| `tests/test_retraining.py` | 3-signal retraining trigger logic |
| `tests/test_drift_checks.py` | PSI computation, ALERT/WARN/OK |
| `tests/test_config.py` | All 19 UC table paths, thresholds |
| `tests/test_quality_checks.py` | Row count, null rate, value range, schema checks (PySpark) |
| `tests/integration_test.py` | 5-stage integration test (runs in preprod) |

### Fixtures (Test Data)
| File | Purpose |
|---|---|
| `fixtures/savingscashflow/base_data_training.csv` | Training fixture data |
| `fixtures/savingscashflow/base_data_inference.csv` | Inference fixture data |
| `fixtures/savingscashflow/moneyfacts_best_buy.csv` | Best-buy rates fixture |
| `fixtures/savingscashflow/qrm_products.csv` | QRM product mapping fixture |
| `fixtures/generate_fixtures.py` | Script to regenerate fixtures |

---

## UC Table Inventory (16 tables + 3 Feature Store tables)

### savings_cashflow schema
| Table | Layer | Written by |
|---|---|---|
| raw_base_data | Bronze | 01_data_ingestion |
| raw_moneyfacts_best_buy | Bronze | 01_data_ingestion |
| raw_qrm_products | Bronze | 01_data_ingestion |
| silver_agg_cohort | Silver | training/02_data_preprocessing |
| inference_features | Silver | inference/02_data_preprocessing |
| inference_predictions | Gold | inference/03_inference (with APE columns) |
| model_eval_metrics | Gold | training/04_model_evaluation |
| monitoring_metrics | Monitoring | inference/04_model_monitoring |
| drift_results | Monitoring | inference/05_data_drift |
| data_quality_results | Monitoring | 02b_data_validation + 06_data_quality |
| dataset_versions | Audit | dataset_versioning.py |
| training_feature_snapshots | Audit | training/02_data_preprocessing |
| model_approvals | Audit | 04b_model_validation + 05a_fmc_validation_gate |
| model_cards | Audit | 05_model_registration |
| deployment_history | Audit | 07_deploy_serving_endpoint |
| rollback_events | Audit | 09_auto_rollback_guard |
| reference_feature_snapshot | Reference | inference/05_data_drift |

### feature_store schema
| Table | Primary Key | Written by |
|---|---|---|
| feature_store_cohort | cohort × product × months_since_start | training/02_data_preprocessing |
| sales_features | account_id | 00_feature_engineering_redshift |
| feature_store_enriched | account_id | 00_feature_engineering_redshift |

### ml_models schema
| Resource | Aliases | Written by |
|---|---|---|
| scf_cohort_model (UC Model Registry) | @baseline, @challenger, @champion | 05_model_registration |

---

## Model Summary

**Model type:** 3 GLMs per product × 10 products = **30 GLMs total**

| GLM | Formula | Family |
|---|---|---|
| Receipts proportion | `rec_prop ~ cr(months_since_start, df=4) + delta_to_best_buy` | Gamma / Tweedie (log link) |
| Outflow proportion | `outflow_prop ~ cr(months_since_start, df=4) + delta_to_best_buy` | Binomial (logit link) |
| Withdrawal split | `withdrawal_prop_of_outflow ~ delta_to_best_buy` | Binomial (logit link) |

**Hyperparameters tuned:** `month_end` ∈ {36,48,50,60} × `drop_month2` ∈ {True,False} = 5 configs per product

**Accuracy (Oct 2025–Feb 2026):** Balance MAPE = **2.28%** on £9–11bn portfolio

- Bundle variable: databricks.yml (notification_webhook_url)

7. Unit and Integration Tests
- Existing unit tests: tests/test_glm_core.py
- Added trigger tests: tests/test_retraining.py

8. CI/CD Gates
- GitHub Actions pipeline: .github/workflows/ci.yml
- Gates: dependency install, pytest, databricks bundle validate -t dev

9. Observability Dashboards
- SQL starter pack for dashboard tiles and alerts: src/dev/observability_dashboard_queries.sql

## Deployment Notes

- Validate:
  databricks bundle validate -t dev
- Deploy:
  databricks bundle deploy -t dev --auto-approve
- If local terraform bootstrap fails, clear local state and redeploy:
  Remove-Item .databricks\bundle\dev -Recurse -Force
  databricks bundle deploy -t dev --auto-approve
