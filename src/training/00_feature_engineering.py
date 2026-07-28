# Databricks notebook source
# MAGIC %md
# MAGIC # 00 - Feature Engineering (GLM flow-ratio lookups)
# MAGIC
# MAGIC The SCF model is a **Generalised Linear Model (GLM)**. It does **not**
# MAGIC predict monthly balance/receipts/withdrawals/transfers directly — it
# MAGIC predicts three **intermediate flow ratios** that are later combined to
# MAGIC reconstruct balances and flows per cohort, then summed to
# MAGIC `product × month`.
# MAGIC
# MAGIC This step reads the cohort matrix produced by `02_data_preprocessing.py`
# MAGIC (`silver_agg_cohort`), validates the ratio ranges the model requires, and
# MAGIC builds the **3 lookup tables** used to project flows:
# MAGIC
# MAGIC | Lookup | Ratio | Indexed by |
# MAGIC |---|---|---|
# MAGIC | `receipts_ratio_lookup`  | `rec_prop`                     | cohort age × rate bucket |
# MAGIC | `outflows_ratio_lookup`  | `outflow_prop`                 | cohort age × rate bucket |
# MAGIC | `transfers_ratio_lookup` | transfer share (1 − wd_prop)   | rate bucket only |
# MAGIC
# MAGIC **Range checks (model-owner requirement):** along with `outflow_prop`,
# MAGIC `rec_prop` (≥ 0, log-link) and `withdrawal_prop_of_outflow` (∈ [0, 1],
# MAGIC binomial) are validated here before the ratios feed the GLM.

# COMMAND ----------
dbutils.widgets.text("catalog", "pd_dtl_ds")
dbutils.widgets.text("schema", "savings_cashflow")
dbutils.widgets.text("feature_schema", "feature_store")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
feature_schema = dbutils.widgets.get("feature_schema")

import sys
sys.path.append("../..")
from src.common.config import get_config
import pyspark.sql.functions as F

cfg = get_config(catalog, schema, feature_schema=feature_schema)
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{feature_schema}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1 — Read the cohort matrix

# COMMAND ----------
agg = spark.table(cfg.silver_agg_cohort)
print(f"silver_agg_cohort rows: {agg.count():,}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2 — Range checks on the GLM flow ratios
# MAGIC The three GLMs assume:
# MAGIC   * `rec_prop`                  ≥ 0     (Gamma/Tweedie log-link)
# MAGIC   * `outflow_prop`              ∈ [0,1] (Binomial)
# MAGIC   * `withdrawal_prop_of_outflow`∈ [0,1] (Binomial)
# MAGIC Out-of-range values are counted, reported, and clamped so a single dirty
# MAGIC cohort can't blow up the fit.

# COMMAND ----------
RANGE_SPECS = {
    "rec_prop": (0.0, None),
    "outflow_prop": (0.0, 1.0),
    "withdrawal_prop_of_outflow": (0.0, 1.0),
}

for col, (lo, hi) in RANGE_SPECS.items():
    cond = F.lit(False)
    if lo is not None:
        cond = cond | (F.col(col) < F.lit(lo))
    if hi is not None:
        cond = cond | (F.col(col) > F.lit(hi))
    n_bad = agg.filter(cond).count()
    print(f"  {col}: {n_bad} rows outside [{lo}, {hi}]")
    if lo is not None:
        agg = agg.withColumn(col, F.greatest(F.col(col), F.lit(lo)))
    if hi is not None:
        agg = agg.withColumn(col, F.least(F.col(col), F.lit(hi)))

# Transfer share is the complement of the withdrawal split.
agg = agg.withColumn(
    "transfer_prop_of_outflow", F.lit(1.0) - F.col("withdrawal_prop_of_outflow")
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3 — Build the 3 lookup tables

# COMMAND ----------
# Receipts & outflows: indexed by cohort age (months_since_start) × rate bucket.
receipts_lookup = (
    agg.groupBy("months_since_start", "dbb_range")
    .agg(F.mean("rec_prop").alias("rec_prop"), F.count("*").alias("n_obs"))
)
outflows_lookup = (
    agg.groupBy("months_since_start", "dbb_range")
    .agg(F.mean("outflow_prop").alias("outflow_prop"), F.count("*").alias("n_obs"))
)

# Transfers: indexed by rate bucket only.
transfers_lookup = (
    agg.groupBy("dbb_range")
    .agg(
        F.mean("transfer_prop_of_outflow").alias("transfer_prop_of_outflow"),
        F.count("*").alias("n_obs"),
    )
)

print(f"receipts_lookup rows:  {receipts_lookup.count():,}")
print(f"outflows_lookup rows:  {outflows_lookup.count():,}")
print(f"transfers_lookup rows: {transfers_lookup.count():,}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4 — Persist lookup tables (Feature Store when available)

# COMMAND ----------
def _write(df, name, pks, desc):
    try:
        from databricks.feature_engineering import FeatureEngineeringClient
        fe = FeatureEngineeringClient()
        try:
            fe.create_table(name=name, primary_keys=pks, df=df, description=desc)
            print(f"Feature table created: {name}")
        except Exception:
            fe.write_table(name=name, df=df, mode="merge")
            print(f"Feature table merged: {name}")
    except ImportError:
        df.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(name)
        print(f"Feature Engineering SDK unavailable — wrote plain Delta: {name}")


_write(
    receipts_lookup, cfg.receipts_lookup,
    ["months_since_start", "dbb_range"],
    "GLM receipts ratio (rec_prop) indexed by cohort age × rate bucket.",
)
_write(
    outflows_lookup, cfg.outflows_lookup,
    ["months_since_start", "dbb_range"],
    "GLM outflow ratio (outflow_prop) indexed by cohort age × rate bucket.",
)
_write(
    transfers_lookup, cfg.transfers_lookup,
    ["dbb_range"],
    "GLM transfer share (1 - withdrawal_prop_of_outflow) indexed by rate bucket.",
)

# COMMAND ----------
assert receipts_lookup.count() > 0, "receipts_lookup empty - aborting pipeline"
assert outflows_lookup.count() > 0, "outflows_lookup empty - aborting pipeline"
assert transfers_lookup.count() > 0, "transfers_lookup empty - aborting pipeline"
print("Feature engineering complete.")
