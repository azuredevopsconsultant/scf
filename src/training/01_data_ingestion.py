# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Data Ingestion (Training)
# MAGIC Reads from CSV fixtures checked into this repo under `fixtures/savingscashflow/`
# MAGIC (synced to the workspace via Databricks Repos / DAB file sync) rather than
# MAGIC S3, so the pipeline runs end-to-end without needing the real PII extract.
# MAGIC Swap `read_source()` for `spark.read.parquet("s3://...")` once the real
# MAGIC S3 paths are available - everything downstream is unaffected either way,
# MAGIC since both paths land in the same `raw_base_data` / `raw_best_buy` /
# MAGIC `raw_qrm_products` UC tables.

# COMMAND ----------
dbutils.widgets.text("catalog", "pd_dtl_ds")
dbutils.widgets.text("schema", "savings_cashflow")
dbutils.widgets.text("bundle_root", "")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
# bundle_root is injected by the job (see resources/training_job.yml) as
# ${workspace.file_path} - the absolute Workspace path this bundle was
# deployed to. Falls back to a relative path for ad-hoc interactive runs
# from within the repo.
bundle_root = dbutils.widgets.get("bundle_root")

import os
import sys
sys.path.append("../..")
from src.common.config import get_config

cfg = get_config(catalog, schema)
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")

if bundle_root:
    FIXTURES_DIR = f"{bundle_root}/fixtures/savingscashflow"
else:
    FIXTURES_DIR = os.path.abspath(os.path.join(os.getcwd(), "..", "..", "fixtures", "savingscashflow"))

print(f"Reading fixtures from: {FIXTURES_DIR}")

# COMMAND ----------
BASE_DATA_PATH = f"{FIXTURES_DIR}/base_data_training.csv"
BEST_BUY_PATH = f"{FIXTURES_DIR}/moneyfacts_best_buy.csv"
QRM_PRODUCTS_PATH = f"{FIXTURES_DIR}/qrm_products.csv"

# COMMAND ----------
base_df = spark.read.option("header", True).option("inferSchema", True).csv(f"file:{BASE_DATA_PATH}")
(
    base_df.write.mode("overwrite")
    .option("mergeSchema", "true")
    .saveAsTable(cfg.raw_base_data)
)
print(f"raw_base_data rows: {base_df.count()}")

# COMMAND ----------
best_buy_df = spark.read.option("header", True).option("inferSchema", True).csv(f"file:{BEST_BUY_PATH}")
best_buy_df.write.mode("overwrite").saveAsTable(cfg.raw_best_buy)
print(f"raw_best_buy rows: {best_buy_df.count()}")

# COMMAND ----------
qrm_df = spark.read.option("header", True).option("inferSchema", True).csv(f"file:{QRM_PRODUCTS_PATH}")
qrm_df.write.mode("overwrite").saveAsTable(cfg.raw_qrm_products)
print(f"raw_qrm_products rows: {qrm_df.count()}")

# COMMAND ----------
# Fail fast if source landed empty - surfaces as a task failure -> job-level
# on_failure email, rather than silently training on stale/empty data.
assert base_df.count() > 0, "raw_base_data landed with 0 rows - aborting pipeline"
assert best_buy_df.count() > 0, "raw_best_buy landed with 0 rows - aborting pipeline"
assert qrm_df.count() > 0, "raw_qrm_products landed with 0 rows - aborting pipeline"

print("Data ingestion complete.")
