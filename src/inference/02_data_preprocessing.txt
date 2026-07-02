# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Data Preprocessing (Inference)
# MAGIC Identical feature pipeline to training (same merges, `months_since_start`,
# MAGIC cohort aggregation, `delta_to_best_buy`/`dbb_range` binning). Keeping
# MAGIC train/inference preprocessing as near-duplicates that both call the same
# MAGIC bucketing constants (`src/common/glm_core.RATE_BINS/RATE_LABELS`) is what
# MAGIC prevents train/serve skew here, since there's no sklearn Pipeline object
# MAGIC to persist - the "features" are the pandas transforms themselves.

# COMMAND ----------
dbutils.widgets.text("catalog", "pd_dtl_ds")
dbutils.widgets.text("schema", "savings_cashflow")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

import sys
sys.path.append("../..")
from src.common.config import get_config
from src.common.dataset_versioning import record_dataset_version
from src.common.glm_core import RATE_BINS, RATE_LABELS
import pandas as pd

cfg = get_config(catalog, schema)

# COMMAND ----------
df = spark.table(cfg.raw_base_data).toPandas()
best_buy = spark.table(cfg.raw_best_buy).toPandas()
prod_df = spark.table(cfg.raw_qrm_products).toPandas()

best_buy_mapping = {
    'Variable ISA (Multi-channel)': 'EA ISA (Multi-Channel)',
    'Variable ISA (Online)': 'EA ISA (Online)',
    'Variable NonISA (Multi-channel)': 'EA Non-ISA (Multi-Channel)',
    'Variable NonISA (Online)': 'EA Non-ISA (Online)',
    'HL ISA': 'EA ISA (Online)',
    'HL NonISA': 'EA Non-ISA (Online)',
    'Money Management': 'EA Non-ISA (Multi-Channel)',
    'Privilege ISA': 'EA Non-ISA (Online)',
    'Restricted ISA': 'Restricted ISA',
    'Restricted NonISA': 'Restricted Non-ISA',
}

df['reporting period'] = pd.to_datetime(df['reporting period']).dt.to_period('M')
best_buy['period'] = pd.to_datetime(best_buy['period']).dt.to_period('M')
df['cohort'] = pd.to_datetime(df['opening date']).dt.to_period('M')

df = df.merge(prod_df, left_on='product code', right_on='Product Code', how='left')
df['best_buy_product_category'] = df['QRM Category'].map(best_buy_mapping)
df = df.merge(
    best_buy, left_on=['reporting period', 'best_buy_product_category'],
    right_on=['period', 'mapping_category'], how='left',
)

bb_end = best_buy['period'].max()
df = df[df['reporting period'] <= bb_end]
df = df[df['QRM Category'].notnull() & (df['QRM Category'] != 'default unassigned')]

df['months_since_start'] = (
    (df['reporting period'].dt.year - df['cohort'].dt.year) * 12
    + (df['reporting period'].dt.month - df['cohort'].dt.month) + 1
)
df['cohort_id'] = df['cohort']

agg_df = (
    df.groupby(['cohort_id', 'QRM Category', 'months_since_start'])
    .agg({
        'reporting period': 'max', 'period balance': 'sum', '£ withdrawal': 'sum',
        '£ receipt': 'sum', '£ internal transfer': 'sum', 'interest rate': 'mean',
        'best_buy': 'mean', 'account tenure (months)': 'mean', 'account id': 'nunique',
    })
    .reset_index()
)
agg_df.columns = [
    'cohort', 'product', 'months_since_start', 'reporting_period', 'balance',
    'withdrawals', 'receipts', 'transfers', 'int_rate', 'best_buy',
    'tenure_months', '#accounts',
]
agg_df = agg_df.sort_values(['cohort', 'product', 'months_since_start'])

agg_df['balance_lag_1'] = agg_df.groupby(['cohort', 'product'])['balance'].shift(1)
agg_df.loc[agg_df['balance_lag_1'] == 0, 'balance_lag_1'] = 1
agg_df['delta_to_best_buy'] = agg_df['int_rate'] - agg_df['best_buy']
agg_df['dbb_range'] = pd.cut(agg_df['delta_to_best_buy'], bins=RATE_BINS, labels=RATE_LABELS)
agg_df = agg_df[~(agg_df['product'] == 'default unassigned')]
agg_df = agg_df.drop_duplicates(['product', 'cohort', 'months_since_start'])

# COMMAND ----------
agg_df['reporting_period'] = agg_df['reporting_period'].astype(str)
agg_df['cohort'] = agg_df['cohort'].astype(str)

spark.createDataFrame(agg_df).write.mode("overwrite").option(
    "mergeSchema", "true"
).saveAsTable(cfg.inference_features)

record_dataset_version(
    spark=spark,
    source_table=cfg.inference_features,
    tracking_table=cfg.dataset_versions,
    dataset_name="inference_features",
    environment=catalog,
)

print(f"inference_features rows: {len(agg_df)}")
assert len(agg_df) > 0, "Inference preprocessing produced 0 rows - aborting pipeline"
