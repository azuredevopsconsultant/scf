# Databricks notebook source
# MAGIC %md
# MAGIC # 02b - Data Validation (Training)
# MAGIC Runs quality gates on `silver_agg_cohort` before model training starts.
# MAGIC Checks: row count floor, null rates on model features, value ranges for
# MAGIC proportions, required schema columns, and minimum product coverage.
# MAGIC Fails the task (job-level on_failure email) if any hard gate fails,
# MAGIC so a bad preprocessing run never silently reaches model training.

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
    check_value_range,
)
import pandas as pd

cfg = get_config(catalog, schema)

# COMMAND ----------
silver_df = spark.table(cfg.silver_agg_cohort)

REQUIRED_COLUMNS = [
    'cohort', 'product', 'months_since_start', 'reporting_period',
    'balance', 'withdrawals', 'receipts', 'transfers',
    'int_rate', 'best_buy', 'tenure_months', '#accounts',
    'balance_lag_1', 'rec_prop', 'outflow_prop',
    'withdrawal_prop_of_outflow', 'delta_to_best_buy', 'dbb_range',
]

MIN_ROWS = 500   # fail if cohort aggregation collapses below this
MAX_NULL_RATE = 0.05  # 5% null ceiling on model features

FEATURE_COLUMNS = [
    'balance', 'int_rate', 'best_buy', 'balance_lag_1',
    'rec_prop', 'outflow_prop', 'delta_to_best_buy',
]

# COMMAND ----------
results = []

# Schema check - must be first so downstream checks don't KeyError
results.append(check_schema_columns(silver_df, REQUIRED_COLUMNS))

# Row count
results.append(check_row_count(silver_df, MIN_ROWS))

# Null rates on model features
results.extend(check_null_rates(silver_df, FEATURE_COLUMNS, MAX_NULL_RATE))

# Proportion ranges: outflow_prop and withdrawal_prop_of_outflow are true
# proportions in [0, 1]; rec_prop can exceed balance so it uses a wider ceiling.
results.append(check_value_range(silver_df, "outflow_prop", 0.0, 1.0))
results.append(check_value_range(silver_df, "rec_prop", 0.0, 5.0))  # receipts can exceed balance
results.append(check_value_range(silver_df, "withdrawal_prop_of_outflow", 0.0, 1.0))

# Product coverage: need at least 5 products to train a meaningful portfolio
pdf = silver_df.select("product").distinct().toPandas()
n_products = len(pdf)
results.append({
    "check": "product_coverage",
    "passed": n_products >= 5,
    "detail": f"{n_products} distinct products (expected >= 5)",
})

# COMMAND ----------
validation_df = pd.DataFrame(results)
validation_df["validated_at"] = datetime.datetime.utcnow().isoformat()
validation_df["pipeline"] = "training"

spark.createDataFrame(validation_df.astype(str)).write.mode("append").option(
    "mergeSchema", "true"
).saveAsTable(cfg.data_quality_results)

# COMMAND ----------
failed = validation_df[~validation_df["passed"].astype(str).str.lower().eq("true")]
if not failed.empty:
    print("DATA VALIDATION FAILURES:")
    for _, row in failed.iterrows():
        print(f"  FAIL  {row['check']}: {row['detail']}")
    raise AssertionError(
        f"{len(failed)} validation check(s) failed — aborting training. "
        "See data_quality_results table for details."
    )

print(f"All {len(results)} data validation checks passed.")
print(f"Silver table: {silver_df.count()} rows, {n_products} products ready for training.")
