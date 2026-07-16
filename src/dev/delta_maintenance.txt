# Databricks notebook source
# MAGIC %md
# MAGIC # Delta Table Maintenance — OPTIMIZE + ZORDER
# MAGIC
# MAGIC Runs OPTIMIZE and ZORDER on the high-read tables to keep query
# MAGIC performance production-grade. Also runs ANALYZE to refresh statistics
# MAGIC for the Unity Catalog query optimizer.
# MAGIC
# MAGIC **Run:** Weekly (scheduled separately, not part of training/inference pipelines).
# MAGIC **Cost:** Reads and rewrites data files — run during off-peak hours.

# COMMAND ----------
dbutils.widgets.text("catalog", "pd_dtl_ds")
dbutils.widgets.text("schema",  "savings_cashflow")
catalog = dbutils.widgets.get("catalog")
schema  = dbutils.widgets.get("schema")

import sys
sys.path.append("../..")
from src.common.config import get_config
import datetime

cfg = get_config(catalog, schema)
start = datetime.datetime.utcnow()
print(f"Delta maintenance started: {start.isoformat()}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## High-read tables: OPTIMIZE + ZORDER

# COMMAND ----------
OPTIMIZE_JOBS = [
    # (table, zorder_cols)  — zorder on the columns most used in WHERE/JOIN
    (cfg.inference_predictions,       ["product", "scored_at"]),
    (cfg.monitoring_metrics,          ["product", "logged_at"]),
    (cfg.drift_results,               ["feature", "run_timestamp"]),
    (cfg.silver_agg_cohort,           ["product", "cohort", "reporting_period"]),
    (cfg.feature_store_table,         ["product", "cohort"]),
    (cfg.training_feature_snapshots,  ["snapshot_timestamp"]),
]

for table, zcols in OPTIMIZE_JOBS:
    zorder_expr = ", ".join(zcols)
    try:
        spark.sql(f"OPTIMIZE {table} ZORDER BY ({zorder_expr})")
        print(f"OPTIMIZE ZORDER done: {table}")
    except Exception as e:
        print(f"WARNING: {table} OPTIMIZE failed: {e}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Audit tables: OPTIMIZE only (no ZORDER — append-only, infrequent)

# COMMAND ----------
OPTIMIZE_ONLY = [
    cfg.dataset_versions,
    cfg.model_approvals,
    cfg.model_cards,
    cfg.deployment_history,
    cfg.rollback_events,
    cfg.data_quality_results,
    cfg.eval_metrics,
]

for table in OPTIMIZE_ONLY:
    try:
        spark.sql(f"OPTIMIZE {table}")
        print(f"OPTIMIZE done: {table}")
    except Exception as e:
        print(f"WARNING: {table} OPTIMIZE failed: {e}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## ANALYZE TABLE — refresh statistics for UC query optimizer

# COMMAND ----------
ANALYZE_TABLES = [
    cfg.silver_agg_cohort,
    cfg.inference_predictions,
    cfg.feature_store_table,
]
for table in ANALYZE_TABLES:
    try:
        spark.sql(f"ANALYZE TABLE {table} COMPUTE STATISTICS FOR ALL COLUMNS")
        print(f"ANALYZE done: {table}")
    except Exception as e:
        print(f"WARNING: {table} ANALYZE failed: {e}")

# COMMAND ----------
elapsed = (datetime.datetime.utcnow() - start).seconds
print(f"\nDelta maintenance complete in {elapsed}s")
