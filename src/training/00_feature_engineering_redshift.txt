# Databricks notebook source
# MAGIC %md
# MAGIC # Feature Engineering — Redshift Sales Features
# MAGIC
# MAGIC Reads sales transaction data from Redshift, computes aggregate features,
# MAGIC and writes them to the Databricks Feature Store.
# MAGIC
# MAGIC **Features created:**
# MAGIC | Feature | Description |
# MAGIC |---|---|
# MAGIC | `sum_of_sales` | Total £ sales per join key |
# MAGIC | `avg_sale_amount` | Average transaction value |
# MAGIC | `sale_count` | Number of transactions |
# MAGIC | `days_since_last_sale` | Recency — days since last transaction |
# MAGIC | `sale_std` | Volatility of sales amounts |
# MAGIC | `max_single_sale` | Largest single transaction |
# MAGIC | `monthly_sale_avg` | Average monthly sales volume |
# MAGIC
# MAGIC These are joined to the existing 3 features already in the Feature Store.

# COMMAND ----------
dbutils.widgets.text("catalog",            "pd_dtl_ds_dev")
dbutils.widgets.text("schema",             "savings_cashflow")
dbutils.widgets.text("feature_schema",     "feature_store")
dbutils.widgets.text("redshift_host",      "")
dbutils.widgets.text("redshift_port",      "5439")
dbutils.widgets.text("redshift_db",        "")
dbutils.widgets.text("redshift_schema",    "sales")
dbutils.widgets.text("redshift_table",     "transactions")
dbutils.widgets.text("join_key",           "account_id")
dbutils.widgets.text("secret_scope",       "scf-cohort-dev")
dbutils.widgets.text("as_of_date",         "")

catalog         = dbutils.widgets.get("catalog")
schema          = dbutils.widgets.get("schema")
feature_schema  = dbutils.widgets.get("feature_schema")
redshift_host   = dbutils.widgets.get("redshift_host")
redshift_port   = dbutils.widgets.get("redshift_port")
redshift_db     = dbutils.widgets.get("redshift_db")
redshift_schema = dbutils.widgets.get("redshift_schema")
redshift_table  = dbutils.widgets.get("redshift_table")
join_key        = dbutils.widgets.get("join_key")
secret_scope    = dbutils.widgets.get("secret_scope")
as_of_date      = dbutils.widgets.get("as_of_date") or None

import sys, datetime
sys.path.append("../..")
from src.common.config import get_config
from databricks.feature_engineering import FeatureEngineeringClient
import pyspark.sql.functions as F

cfg = get_config(catalog, schema, feature_schema=feature_schema)
fe  = FeatureEngineeringClient()

# Create feature_store schema if not exists
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{feature_schema}")

cutoff = as_of_date or datetime.date.today().isoformat()
print(f"Feature engineering as-of: {cutoff}")
print(f"Feature tables → {catalog}.{feature_schema}.*")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1 — Read from Redshift

# COMMAND ----------
# Credentials pulled from Databricks secret scope — never hardcoded
rs_user     = dbutils.secrets.get(secret_scope, "redshift-user")
rs_password = dbutils.secrets.get(secret_scope, "redshift-password")

jdbc_url = (
    f"jdbc:redshift://{redshift_host}:{redshift_port}/{redshift_db}"
    f"?user={rs_user}&password={rs_password}&ssl=true"
)

print(f"Reading {redshift_schema}.{redshift_table} from Redshift...")
sales_sdf = (
    spark.read
    .format("jdbc")
    .option("url",      jdbc_url)
    .option("dbtable",  f"{redshift_schema}.{redshift_table}")
    .option("driver",   "com.amazon.redshift.jdbc42.Driver")
    .option("numPartitions", 8)
    .option("fetchsize", 10000)
    .load()
)

# Filter to as-of date (point-in-time correct features — no data leakage)
if "sale_date" in sales_sdf.columns:
    sales_sdf = sales_sdf.filter(F.col("sale_date") <= cutoff)

print(f"Rows loaded: {sales_sdf.count():,}")
print(f"Columns: {sales_sdf.columns}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2 — Compute Aggregate Features

# COMMAND ----------
today = F.lit(cutoff).cast("date")

features_sdf = (
    sales_sdf
    .groupBy(join_key)
    .agg(
        F.sum("amount").alias("sum_of_sales"),
        F.avg("amount").alias("avg_sale_amount"),
        F.count("*").alias("sale_count"),
        F.stddev("amount").alias("sale_std"),
        F.max("amount").alias("max_single_sale"),
        F.max("sale_date").alias("last_sale_date"),
        F.min("sale_date").alias("first_sale_date"),
    )
    .withColumn(
        "days_since_last_sale",
        F.datediff(today, F.col("last_sale_date"))
    )
    .withColumn(
        "tenure_days",
        F.datediff(F.col("last_sale_date"), F.col("first_sale_date"))
    )
    .withColumn(
        "monthly_sale_avg",
        F.when(F.col("tenure_days") > 0,
               F.col("sum_of_sales") / (F.col("tenure_days") / 30.0)
        ).otherwise(F.col("sum_of_sales"))
    )
    .withColumn("feature_date", F.lit(cutoff))
    # Null-safe defaults
    .fillna({"sale_std": 0.0, "days_since_last_sale": 9999})
)

print(f"Features computed for {features_sdf.count():,} {join_key}s")
display(features_sdf.limit(5))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3 — Write to Feature Store (merge on join_key)

# COMMAND ----------
try:
    fe.create_table(
        name=cfg.sales_features,
        primary_keys=[join_key],
        df=features_sdf,
        description=(
            f"Sales aggregate features computed from Redshift {redshift_schema}.{redshift_table}. "
            f"Primary key: {join_key}. Updated as-of: {cutoff}."
        ),
    )
    print(f"Feature table created: {cfg.sales_features}")
except Exception:
    fe.write_table(name=cfg.sales_features, df=features_sdf, mode="merge")
    print(f"Feature table merged: {cfg.sales_features} ({features_sdf.count():,} rows)")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4 — Join with Existing Feature Store Table

# COMMAND ----------
# Enrich the existing 3 features with the new sales features
# Result is stored in a combined feature table for model training
existing_features = spark.table(cfg.feature_store_cohort)
sales_features    = spark.table(cfg.sales_features)

enriched_sdf = existing_features.join(
    sales_features.drop("feature_date"),
    on=join_key,
    how="left"
)

try:
    fe.create_table(
        name=cfg.feature_store_enriched,
        primary_keys=[join_key],
        df=enriched_sdf,
        description=(
            f"Enriched feature table: existing 3 features + Redshift sales aggregates. "
            f"Used for model training. Updated as-of: {cutoff}."
        ),
    )
except Exception:
    fe.write_table(name=cfg.feature_store_enriched, df=enriched_sdf, mode="merge")

print(f"Enriched feature table: {cfg.feature_store_enriched}")
print(f"Total features: {len(enriched_sdf.columns) - 1}")  # -1 for key

# COMMAND ----------
# Feature summary for MLflow logging
from pyspark.sql.functions import count, col
summary = features_sdf.agg(
    F.sum("sum_of_sales").alias("total_portfolio_sales"),
    F.avg("avg_sale_amount").alias("portfolio_avg_transaction"),
    F.sum("sale_count").alias("total_transactions"),
    F.avg("days_since_last_sale").alias("avg_recency_days"),
).collect()[0]

print(f"\nPortfolio summary:")
print(f"  Total sales      : £{summary['total_portfolio_sales']:,.0f}")
print(f"  Avg transaction  : £{summary['portfolio_avg_transaction']:.2f}")
print(f"  Total tx count   : {summary['total_transactions']:,}")
print(f"  Avg recency days : {summary['avg_recency_days']:.0f}")

dbutils.jobs.taskValues.set(key="feature_table",    value=cfg.sales_features)
dbutils.jobs.taskValues.set(key="enriched_table",   value=cfg.feature_store_enriched)
dbutils.jobs.taskValues.set(key="n_entities",       value=int(features_sdf.count()))
dbutils.jobs.taskValues.set(key="feature_date",     value=cutoff)
