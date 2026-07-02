# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Data Ingestion (Inference)
# MAGIC Reads the "latest batch" fixture (`base_data_inference.csv`) from this
# MAGIC repo instead of S3 - see `src/training/01_data_ingestion.py` for the
# MAGIC training-side equivalent and the rationale. Swap `INFERENCE_DATA_PATH`
# MAGIC for the real incoming S3 extract once available.

# COMMAND ----------
dbutils.widgets.text("catalog", "pd_dtl_ds")
dbutils.widgets.text("schema", "savings_cashflow")
dbutils.widgets.text("bundle_root", "")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
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

BASE_DATA_PATH = f"{FIXTURES_DIR}/base_data_inference.csv"
BEST_BUY_PATH = f"{FIXTURES_DIR}/moneyfacts_best_buy.csv"
QRM_PRODUCTS_PATH = f"{FIXTURES_DIR}/qrm_products.csv"

# COMMAND ----------
base_df = spark.read.option("header", True).option("inferSchema", True).csv(f"file:{BASE_DATA_PATH}")
base_df.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(cfg.raw_base_data)

best_buy_df = spark.read.option("header", True).option("inferSchema", True).csv(f"file:{BEST_BUY_PATH}")
best_buy_df.write.mode("overwrite").saveAsTable(cfg.raw_best_buy)

qrm_df = spark.read.option("header", True).option("inferSchema", True).csv(f"file:{QRM_PRODUCTS_PATH}")
qrm_df.write.mode("overwrite").saveAsTable(cfg.raw_qrm_products)

assert base_df.count() > 0, "raw_base_data landed with 0 rows - aborting inference run"
print("Inference data ingestion complete.")
