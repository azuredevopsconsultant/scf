# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Model Training
# MAGIC Runs `ModelingPipeline.run_for_product()` (ported to
# MAGIC `src/common/glm_core.py`) for every product, exactly as the original
# MAGIC notebook's "Run GLM Pipeline" cell did in a loop. Results (fitted GLMs +
# MAGIC prediction/composite tables per product) are pickled to a UC Volume so
# MAGIC the next task (evaluation) doesn't need to refit.

# COMMAND ----------
dbutils.widgets.text("catalog", "pd_dtl_ds")
dbutils.widgets.text("schema", "savings_cashflow")
dbutils.widgets.text("cutoff_period", "2025-01")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
cutoff_period = dbutils.widgets.get("cutoff_period")

import sys, pickle, datetime
sys.path.append("../..")
from src.common.config import get_config
from src.common.glm_core import ModelingPipeline

cfg = get_config(catalog, schema)

# COMMAND ----------
agg_df = spark.table(cfg.silver_agg_cohort).toPandas()
agg_df['reporting_period'] = agg_df['reporting_period'].astype('period[M]')
agg_df['cohort'] = agg_df['cohort'].astype('period[M]')

# COMMAND ----------
pipeline = ModelingPipeline(agg_df)
all_results = {}

for product in agg_df['product'].unique():
    print(f"Training GLMs for product: {product}")
    try:
        results = pipeline.run_for_product(
            PRODUCT=product,
            cutoff_period=cutoff_period,
            drop_month2=True,
            month_start=2,
            month_end=50,
            proj_start_period=cutoff_period,
            proj_end_period=str(agg_df['reporting_period'].max()),
            proj_cohort_cutoff='2024-12',
            save_csv=False,
        )
        all_results[product] = results
    except Exception as e:
        # One bad product shouldn't fail the whole training run - log and
        # continue, but surface the failure count so evaluation/registration
        # can decide whether coverage is good enough to register.
        print(f"WARNING: training failed for product={product}: {e}")

assert all_results, "No products trained successfully - aborting pipeline"
print(f"Trained {len(all_results)} / {agg_df['product'].nunique()} products")

# COMMAND ----------
# Persist to a UC Volume so 04_model_evaluation.py / 05_model_registration.py
# can pick up the exact same fitted objects without retraining. UC Volumes
# are addressable as normal filesystem paths on UC-enabled clusters - no
# dbfs:/ prefix or dbutils.fs needed for plain Python file I/O.
VOLUME_PATH = f"/Volumes/{catalog}/{schema}/model_artifacts"
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.model_artifacts")

artifact = {
    "results": all_results,
    "trained_at": datetime.datetime.utcnow().isoformat(),
    "cutoff_period": cutoff_period,
}
artifact_path = f"{VOLUME_PATH}/latest_training_run.pkl"
with open(artifact_path, "wb") as f:
    pickle.dump(artifact, f)

print(f"Saved training artifact to {artifact_path}")
