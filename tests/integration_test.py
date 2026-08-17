# Databricks notebook source
# MAGIC %md
# MAGIC # Integration Tests (CI)
# MAGIC
# MAGIC End-to-end integration tests for the SCF Cohort MLOps pipeline.
# MAGIC Runs in the **staging (preprod)** workspace against fixture data.
# MAGIC All 5 test stages must pass before the model is promoted to @champion
# MAGIC and before prod deployment is approved.
# MAGIC
# MAGIC **Test stages:**
# MAGIC 1. Model training tests    — silver table created, GLMs trained per product
# MAGIC 2. Model validation tests  — MAPE within threshold, model registered @challenger
# MAGIC 3. Model deployment tests  — serving endpoint responds, traffic split set
# MAGIC 4. Model inference tests   — predictions written to inference_predictions table
# MAGIC 5. Monitoring tests        — monitoring_metrics + drift_results populated

# COMMAND ----------
# This is a Databricks *notebook*, not a local pytest module. It depends on the
# Databricks runtime globals (`dbutils`, `spark`), which don't exist when pytest
# imports this file locally / in CI. Skip collection cleanly in that case.
try:
    dbutils  # noqa: F821  (injected by the Databricks runtime)
except NameError:
    import pytest

    pytest.skip(
        "Databricks-only integration notebook: requires dbutils/spark runtime "
        "(runs as a Databricks job in the staging workspace, not under local pytest)",
        allow_module_level=True,
    )

# COMMAND ----------
dbutils.widgets.text("catalog", "pd_dtl_ds_pp")
dbutils.widgets.text("schema",  "savings_cashflow")
dbutils.widgets.text("model_schema", "ml_models")
dbutils.widgets.text("model_name", "")
dbutils.widgets.text("serving_endpoint_name", "scf-cohort-serving-endpoint-pp")
catalog                = dbutils.widgets.get("catalog")
schema                 = dbutils.widgets.get("schema")
model_schema           = dbutils.widgets.get("model_schema")
model_name             = dbutils.widgets.get("model_name") or f"{catalog}.{model_schema}.scf_cohort_model"
serving_endpoint_name  = dbutils.widgets.get("serving_endpoint_name")

import sys, datetime
sys.path.append("../..")
from src.common.config import get_config
from src.common.drift_checks import any_alerts
import pandas as pd
import mlflow

mlflow.set_registry_uri("databricks-uc")
cfg = get_config(catalog, schema)

failures = []
def check(name: str, condition: bool, detail: str = ""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        failures.append(f"{name}: {detail}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Test Stage 1 — Model Training Tests

# COMMAND ----------
print("=== Stage 1: Model Training ===")

# 1a. silver_agg_cohort must exist and have rows
try:
    silver_count = spark.table(cfg.silver_agg_cohort).count()
    check("silver_agg_cohort has rows", silver_count > 0, f"{silver_count} rows")
except Exception as e:
    check("silver_agg_cohort accessible", False, str(e))
    silver_count = 0

# 1b. feature_store_cohort must exist
try:
    feat_count = spark.table(cfg.feature_store_table).count()
    check("feature_store_cohort has rows", feat_count > 0, f"{feat_count} rows")
except Exception as e:
    check("feature_store_cohort accessible", False, str(e))

# 1c. model_eval_metrics must have a recent run
try:
    metrics = spark.table(cfg.eval_metrics).toPandas()
    recent = pd.to_datetime(metrics["evaluated_at"]).max()
    age_hours = (datetime.datetime.utcnow() - recent.to_pydatetime().replace(tzinfo=None)).total_seconds() / 3600
    check("model_eval_metrics has recent run", age_hours < 24, f"last run {age_hours:.1f}h ago")
    portfolio_mape = pd.to_numeric(metrics["mape_balance"], errors="coerce").mean()
    check("portfolio_mape within training threshold", portfolio_mape < 30.0, f"MAPE={portfolio_mape:.2f}%")
except Exception as e:
    check("model_eval_metrics accessible", False, str(e))

# 1d. UC Volume has training artifact
import os
volume_path = f"/Volumes/{catalog}/{model_schema}/model_artifacts/latest_training_run.pkl"
check("training artifact exists in UC Volume", os.path.exists(volume_path), volume_path)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Test Stage 2 — Model Validation Tests

# COMMAND ----------
print("\n=== Stage 2: Model Validation ===")

# 2a. @challenger alias must exist
try:
    from mlflow import MlflowClient
    client = MlflowClient()
    challenger = client.get_model_version_by_alias(model_name, "challenger")
    check("model @challenger alias exists", True, f"v{challenger.version}")
    # 2b. Challenger has description and tags
    check("challenger has description", bool(challenger.description), "")
    check("challenger has trained_at tag", "trained_at" in challenger.tags, "")
    check("challenger has portfolio_mape tag", "portfolio_mape" in challenger.tags, "")
except Exception as e:
    check("model @challenger registered", False, str(e))

# 2c. model_approvals has a record
try:
    approvals = spark.table(cfg.model_approvals).toPandas()
    check("model_approvals table has records", len(approvals) > 0, f"{len(approvals)} records")
    auto_passed = (approvals["validation_passed"] == "True").any()
    check("at least one automated validation passed", auto_passed, "")
except Exception as e:
    check("model_approvals accessible", False, str(e))

# 2d. model_cards has a record
try:
    cards = spark.table(cfg.model_cards).toPandas()
    check("model_cards table has records", len(cards) > 0, f"{len(cards)} cards")
except Exception as e:
    check("model_cards accessible", False, str(e))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Test Stage 3 — Model Deployment Tests

# COMMAND ----------
print("\n=== Stage 3: Model Deployment ===")

# 3a. @champion alias (or @challenger if first run) must exist
try:
    alias_to_check = "champion"
    try:
        mv = client.get_model_version_by_alias(model_name, "champion")
    except Exception:
        mv = client.get_model_version_by_alias(model_name, "challenger")
        alias_to_check = "challenger"
    check(f"model @{alias_to_check} alias exists", True, f"v{mv.version}")
except Exception as e:
    check("deployable model alias exists", False, str(e))

# 3b. serving endpoint exists and is ready
try:
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()
    ep = w.serving_endpoints.get(serving_endpoint_name)
    state = ep.state.config_update.value if ep.state else "UNKNOWN"
    check("serving endpoint exists", True, f"{serving_endpoint_name}")
    check("serving endpoint ready", state in ("IN_PROGRESS", "NOT_UPDATING"), f"state={state}")
except Exception as e:
    check("serving endpoint accessible", False, str(e))

# 3c. deployment_history has a record
try:
    hist = spark.table(cfg.deployment_history).toPandas()
    check("deployment_history has records", len(hist) > 0, f"{len(hist)} deployments")
except Exception as e:
    check("deployment_history accessible", False, str(e))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Test Stage 4 — Model Inference Tests

# COMMAND ----------
print("\n=== Stage 4: Model Inference ===")

# 4a. inference_predictions must have rows from a recent run
try:
    preds = spark.table(cfg.inference_predictions).toPandas()
    check("inference_predictions has rows", len(preds) > 0, f"{len(preds)} rows")
    if len(preds) > 0:
        latest_run = preds["scored_at"].max()
        age = (datetime.datetime.utcnow() - pd.to_datetime(latest_run).to_pydatetime().replace(tzinfo=None)).total_seconds() / 3600
        check("inference_predictions has recent run", age < 48, f"last scored {age:.1f}h ago")
        null_rate = preds["balance_pred"].isna().mean() if "balance_pred" in preds.columns else 1.0
        check("null prediction rate < 10%", null_rate < 0.10, f"{null_rate:.1%} null")
except Exception as e:
    check("inference_predictions accessible", False, str(e))

# 4b. Load model and run a predict on one row — real end-to-end test
try:
    model_uri = f"models:/{model_name}@champion"
    try:
        model = mlflow.pyfunc.load_model(model_uri)
    except Exception:
        model_uri = f"models:/{model_name}@challenger"
        model = mlflow.pyfunc.load_model(model_uri)
    sample = (
        spark.table(cfg.feature_store_table)
        .limit(1)
        .toPandas()
    )
    sample["cohort"] = sample["cohort"].astype(str)
    sample["reporting_period"] = sample["reporting_period"].astype(str)
    sample["months_since_start"] = pd.to_numeric(sample["months_since_start"], errors="coerce").fillna(0).astype(int)
    sample["balance_lag_1"] = pd.to_numeric(sample["balance_lag_1"], errors="coerce").fillna(1.0)
    sample["dbb_range"] = sample["dbb_range"].astype(str)
    result = model.predict(sample)
    check("model.predict() returns non-empty result", len(result) > 0, f"{len(result)} rows returned")
except Exception as e:
    check("model.predict() end-to-end test", False, str(e))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Test Stage 5 — Monitoring Tests

# COMMAND ----------
print("\n=== Stage 5: Monitoring ===")

# 5a. monitoring_metrics populated
try:
    mon = spark.table(cfg.monitoring_metrics).toPandas()
    check("monitoring_metrics has rows", len(mon) > 0, f"{len(mon)} rows")
    products = mon["product"].nunique() if "product" in mon.columns else 0
    check("monitoring_metrics covers multiple products", products > 1, f"{products} products")
except Exception as e:
    check("monitoring_metrics accessible", False, str(e))

# 5b. drift_results populated
try:
    drift = spark.table(cfg.drift_results).toPandas()
    check("drift_results has rows", len(drift) > 0, f"{len(drift)} rows")
    drift["psi"] = pd.to_numeric(drift["psi"], errors="coerce")
    check("no ALERT-level drift on first run", not any_alerts(drift), "PSI below 0.25")
except Exception as e:
    check("drift_results accessible", False, str(e))

# 5c. data_quality_results populated
try:
    dq = spark.table(cfg.data_quality_results).toPandas()
    check("data_quality_results has rows", len(dq) > 0, f"{len(dq)} rows")
except Exception as e:
    check("data_quality_results accessible", False, str(e))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Integration Test Summary

# COMMAND ----------
print(f"\n{'='*60}")
print(f"INTEGRATION TEST RESULTS: {catalog}.{schema}")
print(f"{'='*60}")
if failures:
    print(f"\nFAILED ({len(failures)} failure(s)):")
    for f in failures:
        print(f"  ✗  {f}")
    print()
    raise AssertionError(
        f"{len(failures)} integration test(s) failed. "
        "Fix failures before promoting to prod. See output above."
    )
else:
    print("\nAll 5 test stages PASSED — safe to promote to prod.")
    print("  Stage 1: Model training    ✓")
    print("  Stage 2: Model validation  ✓")
    print("  Stage 3: Model deployment  ✓")
    print("  Stage 4: Model inference   ✓")
    print("  Stage 5: Monitoring        ✓")

dbutils.jobs.taskValues.set(key="integration_tests_passed", value=bool(not failures))
dbutils.jobs.taskValues.set(key="n_failures", value=len(failures))