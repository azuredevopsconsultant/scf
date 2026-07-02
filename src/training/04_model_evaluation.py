# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - Model Evaluation
# MAGIC Aggregates the per-product MAPE/RMSE metrics that `ModelingPipeline`
# MAGIC already computes (same `Evaluator.mape` / `Evaluator.rmse` as the source
# MAGIC notebook) into one portfolio-level metric set, and persists an
# MAGIC auditable metrics table. This is what `model_registration.py` compares
# MAGIC against the current Champion.

# COMMAND ----------
dbutils.widgets.text("catalog", "pd_dtl_ds")
dbutils.widgets.text("schema", "savings_cashflow")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

import sys, pickle, datetime
sys.path.append("../..")
from src.common.config import get_config
import pandas as pd

cfg = get_config(catalog, schema)
VOLUME_PATH = f"/Volumes/{catalog}/{schema}/model_artifacts"

# COMMAND ----------
with open(f"{VOLUME_PATH}/latest_training_run.pkl", "rb") as f:
    artifact = pickle.load(f)

all_results = artifact["results"]

# COMMAND ----------
rows = []
for product, res in all_results.items():
    metrics = res.get("metrics")
    if not metrics:
        print(f"WARNING: no projection/metrics for product={product} (no proj window overlap) - skipped")
        continue
    row = {"product": product, "evaluated_at": datetime.datetime.utcnow().isoformat()}
    row.update(metrics)
    rows.append(row)

metrics_df = pd.DataFrame(rows)
assert not metrics_df.empty, "No products produced evaluation metrics - aborting pipeline"

# Portfolio-level summary metric used for Champion/Challenger comparison:
# balance MAPE, averaged (unweighted) across products. Swap to a
# balance-weighted average if some products dominate total book size.
portfolio_mape = metrics_df["mape_balance"].mean()
portfolio_rmse = metrics_df["rmse_balance"].mean()

print(f"Portfolio MAPE (balance): {portfolio_mape:.2f}%")
print(f"Portfolio RMSE (balance): {portfolio_rmse:.2f}")

# COMMAND ----------
spark.createDataFrame(metrics_df).write.mode("append").option(
    "mergeSchema", "true"
).saveAsTable(cfg.eval_metrics)

# Stash the portfolio metric for the registration task to read via task values
# (cheap way to pass small scalars between notebook tasks in a DAB job -
# avoids a round trip through a table for a single number).
dbutils.jobs.taskValues.set(key="portfolio_mape", value=float(portfolio_mape))
dbutils.jobs.taskValues.set(key="portfolio_rmse", value=float(portfolio_rmse))
dbutils.jobs.taskValues.set(key="n_products_evaluated", value=int(len(metrics_df)))

print(f"Evaluation complete. {cfg.eval_metrics} updated with {len(metrics_df)} rows.")
