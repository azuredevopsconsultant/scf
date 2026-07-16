# Databricks notebook source
# MAGIC %md
# MAGIC # Lakehouse Monitoring Setup
# MAGIC
# MAGIC Enables **Databricks Lakehouse Monitoring** on the key SCF Cohort tables.
# MAGIC Run this notebook once after initial deployment — monitoring profiles
# MAGIC persist and auto-update on every table write.
# MAGIC
# MAGIC **Tables monitored:**
# MAGIC | Table | Profile type | What it tracks |
# MAGIC |---|---|---|
# MAGIC | `inference_predictions` | TimeSeries | Prediction distribution, drift, freshness |
# MAGIC | `monitoring_metrics` | TimeSeries | Live MAPE/RMSE trends over time |
# MAGIC | `feature_store_cohort` | Snapshot | Feature distribution at training time |
# MAGIC | `drift_results` | TimeSeries | PSI trend per feature |

# COMMAND ----------
dbutils.widgets.text("catalog", "pd_dtl_ds")
dbutils.widgets.text("schema",  "savings_cashflow")
catalog = dbutils.widgets.get("catalog")
schema  = dbutils.widgets.get("schema")

import sys
sys.path.append("../..")
from src.common.config import get_config
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    MonitorTimeSeries, MonitorSnapshot,
    MonitorInferenceLog, MonitorInferenceLogProblemType,
)

cfg = get_config(catalog, schema)
w = WorkspaceClient()
assets_dir = f"/Shared/lakehouse_monitoring/{catalog}/{schema}"

print(f"Setting up Lakehouse Monitoring for {catalog}.{schema}")
print(f"Assets directory: {assets_dir}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1 — inference_predictions (TimeSeries)
# MAGIC Tracks prediction volume, balance_pred distribution, null rates,
# MAGIC and model version changes over time. Uses `scored_at` as the timestamp.

# COMMAND ----------
def create_or_update_monitor(table_name, profile, label):
    try:
        w.quality_monitors.create(
            table_name=table_name,
            assets_dir=f"{assets_dir}/{label}",
            output_schema_name=f"{catalog}.{schema}",
            **profile,
        )
        print(f"Created monitor: {table_name}")
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"Monitor already exists: {table_name} — skipping.")
        else:
            print(f"WARNING: Could not create monitor for {table_name}: {e}")

# inference_predictions — TimeSeries on scored_at
create_or_update_monitor(
    table_name=cfg.inference_predictions,
    label="inference_predictions",
    profile={
        "time_series": MonitorTimeSeries(
            timestamp_col="scored_at",
            granularities=["1 day", "1 week"],
        ),
        "slicing_exprs": ["product", "model_version"],
        "skip_builtin_dashboard": False,
    },
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2 — monitoring_metrics (TimeSeries)
# MAGIC Tracks live MAPE, RMSE, and null prediction rate per product over time.

# COMMAND ----------
create_or_update_monitor(
    table_name=cfg.monitoring_metrics,
    label="monitoring_metrics",
    profile={
        "time_series": MonitorTimeSeries(
            timestamp_col="logged_at",
            granularities=["1 day", "1 week", "1 month"],
        ),
        "slicing_exprs": ["product"],
        "skip_builtin_dashboard": False,
    },
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3 — feature_store_cohort (Snapshot)
# MAGIC Monitors feature distributions at each training run.
# MAGIC Detects schema drift and value range violations across training cycles.

# COMMAND ----------
create_or_update_monitor(
    table_name=cfg.feature_store_table,
    label="feature_store_cohort",
    profile={
        "snapshot": MonitorSnapshot(),
        "slicing_exprs": ["product"],
        "skip_builtin_dashboard": False,
    },
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4 — drift_results (TimeSeries)
# MAGIC Tracks PSI trend per feature over time — rising PSI signals
# MAGIC gradual distribution shift before it reaches the ALERT threshold.

# COMMAND ----------
create_or_update_monitor(
    table_name=cfg.drift_results,
    label="drift_results",
    profile={
        "time_series": MonitorTimeSeries(
            timestamp_col="run_timestamp",
            granularities=["1 day", "1 week"],
        ),
        "slicing_exprs": ["feature"],
        "skip_builtin_dashboard": False,
    },
)

# COMMAND ----------
print("\nLakehouse Monitoring setup complete.")
print(f"View dashboards in Databricks UI: Catalog → {catalog}.{schema} → <table> → Quality tab")
print(f"Monitor assets stored at: {assets_dir}")
print("\nTables monitored:")
for t in [cfg.inference_predictions, cfg.monitoring_metrics,
          cfg.feature_store_table, cfg.drift_results]:
    print(f"  {t}")
