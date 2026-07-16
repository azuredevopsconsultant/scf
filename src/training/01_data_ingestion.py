# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Data Ingestion (Training)
# MAGIC Reads real source files from a UC Volume:
# MAGIC   - base_query_df_apr_2026.parquet  -> raw_base_data
# MAGIC   - moneyfacts_merge.parquet        -> raw_best_buy
# MAGIC   - QRM_products_revised.csv        -> raw_qrm_products
# MAGIC Volume path is controlled by the `volume_path` widget. Same source as
# MAGIC the inference pipeline so training and inference land in identical
# MAGIC `raw_base_data` / `raw_best_buy` / `raw_qrm_products` UC tables.

# COMMAND ----------
dbutils.widgets.text("catalog", "pd_dtl_ds")
dbutils.widgets.text("schema", "savings_cashflow")
dbutils.widgets.text("volume_path", "/Volumes/preprod_catalog/ml_workspace/data_scientists/SCF")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
volume_path = dbutils.widgets.get("volume_path")

import sys
sys.path.append("../..")
from src.common.config import get_config

cfg = get_config(catalog, schema)
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")

print(f"Reading source files from: {volume_path}")

# COMMAND ----------
BASE_DATA_PATH = f"{volume_path}/base_query_df_apr_2026.parquet"
BEST_BUY_PATH = f"{volume_path}/moneyfacts_merge.parquet"
QRM_PRODUCTS_PATH = f"{volume_path}/QRM_products_revised.csv"

# COMMAND ----------
base_df = spark.read.parquet(BASE_DATA_PATH)
(
    base_df.write.mode("overwrite")
    .option("mergeSchema", "true")
    .saveAsTable(cfg.raw_base_data)
)
print(f"raw_base_data rows: {base_df.count()}")

# COMMAND ----------
best_buy_df = spark.read.parquet(BEST_BUY_PATH)
best_buy_df.write.mode("overwrite").saveAsTable(cfg.raw_best_buy)
print(f"raw_best_buy rows: {best_buy_df.count()}")

# COMMAND ----------
qrm_df = spark.read.option("header", True).option("inferSchema", True).csv(QRM_PRODUCTS_PATH)
qrm_df.write.mode("overwrite").saveAsTable(cfg.raw_qrm_products)
print(f"raw_qrm_products rows: {qrm_df.count()}")

# COMMAND ----------
# Fail fast if source landed empty - surfaces as a task failure -> job-level
# on_failure email, rather than silently training on stale/empty data.
assert base_df.count() > 0, "raw_base_data landed with 0 rows - aborting pipeline"
assert best_buy_df.count() > 0, "raw_best_buy landed with 0 rows - aborting pipeline"
assert qrm_df.count() > 0, "raw_qrm_products landed with 0 rows - aborting pipeline"

print("Data ingestion complete.")
