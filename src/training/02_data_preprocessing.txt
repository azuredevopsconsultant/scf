# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Data Preprocessing (Training)
# MAGIC Ports the original notebook's "Data Pre-processing" + "Build Cohort
# MAGIC Matrix" sections (cells 6-45): best-buy category mapping, date -> period
# MAGIC conversion, merges, `months_since_start`, cohort-level aggregation,
# MAGIC interim ratio construction (`rec_prop`, `outflow_prop`,
# MAGIC `withdrawal_prop_of_outflow`), and the increasing-cohort-size exclusion
# MAGIC rule. Runs on the driver with pandas (as the original did) since cohort
# MAGIC volumes are small relative to cluster memory; swap to pandas-on-Spark if
# MAGIC that stops being true.

# COMMAND ----------
dbutils.widgets.text("catalog", "pd_dtl_ds")
dbutils.widgets.text("schema", "savings_cashflow")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

import sys
sys.path.append("../..")
from src.common.config import get_config
from src.common.dataset_versioning import record_dataset_version
import pandas as pd
import numpy as np

cfg = get_config(catalog, schema)

# COMMAND ----------
df = spark.table(cfg.raw_base_data).toPandas()
best_buy = spark.table(cfg.raw_best_buy).toPandas()
prod_df = spark.table(cfg.raw_qrm_products).toPandas()

# COMMAND ----------
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

# COMMAND ----------
df = df.merge(prod_df, left_on='product code', right_on='Product Code', how='left')
df['best_buy_product_category'] = df['QRM Category'].map(best_buy_mapping)
df = df.merge(
    best_buy,
    left_on=['reporting period', 'best_buy_product_category'],
    right_on=['period', 'mapping_category'],
    how='left',
)

# Drop the most recent reporting period if best-buy coverage hasn't caught up yet
bb_end = best_buy['period'].max()
df = df[df['reporting period'] <= bb_end]

# Exclude unmapped QRM categories - same rule as the source notebook
df = df[df['QRM Category'].notnull() & (df['QRM Category'] != 'default unassigned')]

# COMMAND ----------
df['months_since_start'] = (
    (df['reporting period'].dt.year - df['cohort'].dt.year) * 12
    + (df['reporting period'].dt.month - df['cohort'].dt.month)
    + 1
)
df['cohort_id'] = df['cohort']

agg_df = (
    df.groupby(['cohort_id', 'QRM Category', 'months_since_start'])
    .agg(
        {
            'reporting period': 'max',
            'period balance': 'sum',
            '£ withdrawal': 'sum',
            '£ receipt': 'sum',
            '£ internal transfer': 'sum',
            'interest rate': 'mean',
            'best_buy': 'mean',
            'account tenure (months)': 'mean',
            'account id': 'nunique',
        }
    )
    .reset_index()
)
agg_df.columns = [
    'cohort', 'product', 'months_since_start', 'reporting_period', 'balance',
    'withdrawals', 'receipts', 'transfers', 'int_rate', 'best_buy',
    'tenure_months', '#accounts',
]
agg_df = agg_df.sort_values(['cohort', 'product', 'months_since_start'])

# COMMAND ----------
agg_df['balance_lag_1'] = agg_df.groupby(['cohort', 'product'])['balance'].shift(1)
agg_df.loc[agg_df['balance_lag_1'] == 0, 'balance_lag_1'] = 1

agg_df['rec_prop'] = agg_df['receipts'] / agg_df['balance_lag_1']
agg_df['outflow_prop'] = (agg_df['withdrawals'] + agg_df['transfers']) / (
    agg_df['receipts'] + agg_df['balance_lag_1']
)
agg_df['withdrawal_prop_of_outflow'] = agg_df['withdrawals'] / (
    agg_df['withdrawals'] + agg_df['transfers']
)

# COMMAND ----------
# Exclude cohort/product pairs whose #accounts increases at any point
# (portfolio growth breaks the outflow_prop <= 1 assumption)
invalid = agg_df.loc[agg_df['outflow_prop'] > 1, ['cohort', 'product']].drop_duplicates()
df_sub = agg_df.merge(invalid, on=['cohort', 'product'], how='inner')
increasing_groups = (
    df_sub.groupby(['cohort', 'product'])['#accounts']
    .apply(lambda s: s.diff().gt(0).any())
    .rename('increasing_size')
    .reset_index()
)
agg_df = agg_df.merge(increasing_groups, on=['cohort', 'product'], how='left')
agg_df = agg_df[~(agg_df['increasing_size'] == True)]  # noqa: E712

# Cap remaining >1 outflow_prop caused by interest adjustments, not growth
agg_df.loc[agg_df['outflow_prop'] > 1, 'outflow_prop'] = 1

# COMMAND ----------
agg_df.loc[agg_df['withdrawal_prop_of_outflow'].isnull(), 'withdrawal_prop_of_outflow'] = 0

# Delta-to-best-buy: how far a cohort's rate sits from the market best-buy rate,
# bucketed into the same RATE_BINS used at inference time (kept in one place -
# src/common/glm_core.py - so train/inference can never drift apart).
from src.common.glm_core import RATE_BINS, RATE_LABELS

agg_df['delta_to_best_buy'] = agg_df['int_rate'] - agg_df['best_buy']
agg_df['dbb_range'] = pd.cut(agg_df['delta_to_best_buy'], bins=RATE_BINS, labels=RATE_LABELS)
agg_df = agg_df[~(agg_df['product'] == 'default unassigned')]
agg_df = agg_df.drop_duplicates(['product', 'cohort', 'months_since_start'])

# COMMAND ----------
agg_df['reporting_period'] = agg_df['reporting_period'].astype(str)
agg_df['cohort'] = agg_df['cohort'].astype(str)

spark.createDataFrame(agg_df).write.mode("overwrite").option(
    "mergeSchema", "true"
).saveAsTable(cfg.silver_agg_cohort)

spark.createDataFrame(agg_df).write.mode("overwrite").option(
    "mergeSchema", "true"
).saveAsTable(cfg.feature_store_table)

record_dataset_version(
    spark=spark,
    source_table=cfg.silver_agg_cohort,
    tracking_table=cfg.dataset_versions,
    dataset_name="training_silver_agg_cohort",
    environment=catalog,
)

print(f"silver_agg_cohort rows: {len(agg_df)}")
assert len(agg_df) > 0, "Preprocessing produced 0 rows - aborting pipeline"
