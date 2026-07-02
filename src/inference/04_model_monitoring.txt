# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - Model Monitoring (Metrics)
# MAGIC Computes the same MAPE/RMSE the training evaluation task used, but on
# MAGIC live inference vs. actuals (once actuals land for the scored period),
# MAGIC plus operational metrics (prediction volume, null-prediction rate).
# MAGIC Feeds a monitoring table a BI dashboard / Databricks SQL alert can sit on
# MAGIC top of.

# COMMAND ----------
dbutils.widgets.text("catalog", "pd_dtl_ds")
dbutils.widgets.text("schema", "savings_cashflow")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

import sys, datetime
sys.path.append("../..")
from src.common.config import get_config
from src.common.glm_core import Evaluator
import pandas as pd

cfg = get_config(catalog, schema)

# COMMAND ----------
preds = spark.table(cfg.inference_predictions).toPandas()
latest_run = preds['scored_at'].max()
latest_preds = preds[preds['scored_at'] == latest_run].copy()

for col in ['balance', 'balance_pred', 'receipts', 'receipts_pred',
            'withdrawals', 'withdrawals_pred', 'transfers', 'transfers_pred']:
    latest_preds[col] = pd.to_numeric(latest_preds[col], errors='coerce')

# COMMAND ----------
evaluator = Evaluator()
rows = []
for product, grp in latest_preds.groupby('product'):
    actual_available = grp['balance'].notna().any()
    row = {
        "product": product,
        "run_timestamp": latest_run,
        "n_predictions": len(grp),
        "null_prediction_rate": grp['balance_pred'].isna().mean(),
        "has_actuals": bool(actual_available),
    }
    if actual_available:
        row["mape_balance"] = evaluator.mape(grp, 'balance', 'balance_pred')
        row["rmse_balance"] = evaluator.rmse(grp, 'balance', 'balance_pred')
    rows.append(row)

monitoring_df = pd.DataFrame(rows)
monitoring_df["logged_at"] = datetime.datetime.utcnow().isoformat()

# COMMAND ----------
spark.createDataFrame(monitoring_df.astype(str)).write.mode("append").option(
    "mergeSchema", "true"
).saveAsTable(cfg.monitoring_metrics)

# Operational guardrail: alert-worthy null-prediction spike fails the task
# outright (job-level on_failure email fires) rather than waiting for the
# weekly retrain to notice.
max_null_rate = monitoring_df["null_prediction_rate"].max()
portfolio_mape = None
if "mape_balance" in monitoring_df.columns:
    mape_series = pd.to_numeric(monitoring_df["mape_balance"], errors="coerce").dropna()
    if not mape_series.empty:
        portfolio_mape = float(mape_series.mean())

dbutils.jobs.taskValues.set(key="max_null_rate", value=float(max_null_rate))
if portfolio_mape is not None:
    dbutils.jobs.taskValues.set(key="portfolio_mape_monitoring", value=float(portfolio_mape))

assert max_null_rate < 0.5, (
    f"Null prediction rate {max_null_rate:.1%} exceeds 50% for at least one product - "
    "likely a broken model_alias load or feature mismatch."
)

print(f"Model monitoring complete: {len(monitoring_df)} products logged.")
