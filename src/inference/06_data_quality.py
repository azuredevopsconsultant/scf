# Databricks notebook source
# MAGIC %md
# MAGIC # 06 - Data Quality
# MAGIC Runs the reusable checks in `src/common/quality_checks.py` against the
# MAGIC feature table used for scoring: row-count floor, null-rate ceilings on
# MAGIC critical columns, valid ranges for the modelled proportions, and schema
# MAGIC presence. Fails the task (-> job-level failure email) if any check
# MAGIC fails, since bad inputs here would otherwise silently propagate into
# MAGIC scored predictions.

# COMMAND ----------
dbutils.widgets.text("catalog", "pd_dtl_ds")
dbutils.widgets.text("schema", "savings_cashflow")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

import sys, datetime
sys.path.append("../..")
from src.common.config import get_config
from src.common.quality_checks import (
    check_row_count, check_null_rates, check_schema_columns,
    check_value_range, all_checks_passed,
)
import pandas as pd

cfg = get_config(catalog, schema)

# COMMAND ----------
features_df = spark.table(cfg.inference_features)

results = []
results.append(check_schema_columns(
    features_df,
    ["product", "cohort", "reporting_period", "months_since_start",
     "balance", "balance_lag_1", "delta_to_best_buy", "dbb_range"],
))
results.append(check_row_count(features_df, expected_min=100))
# Null checks — includes the two columns the model directly uses at score time
results.extend(check_null_rates(
    features_df,
    columns=["balance", "product", "cohort", "balance_lag_1", "dbb_range"],
    max_null_rate=cfg.NULL_RATE_FAIL_THRESHOLD,
))
# delta_to_best_buy expected range: ±3 pp covers all realistic rate environments
results.append(check_value_range(features_df, "delta_to_best_buy", -3.0, 3.0))

quality_df = pd.DataFrame(results)
quality_df["run_timestamp"] = datetime.datetime.utcnow().isoformat()

spark.createDataFrame(quality_df.astype(str)).write.mode("append").option(
    "mergeSchema", "true"
).saveAsTable(cfg.data_quality_results)

# COMMAND ----------
passed = all_checks_passed(results)
dbutils.jobs.taskValues.set(key="data_quality_passed", value=bool(passed))

print(quality_df.to_string(index=False))
assert passed, "One or more data quality checks failed - see data_quality_results table."
print("All data quality checks passed.")
