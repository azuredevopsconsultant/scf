# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - Model Evaluation
# MAGIC Aggregates per-product MAPE/RMSE + **baseline benchmark comparison**.
# MAGIC For each product, computes how much the GLM improves over the naive
# MAGIC last-value forecast (balance_lag_1 as prediction). A product where the
# MAGIC GLM does NOT beat the baseline is flagged in the metrics table.
# MAGIC Portfolio-level metrics passed forward via task values.

# COMMAND ----------
dbutils.widgets.text("catalog", "pd_dtl_ds")
dbutils.widgets.text("schema", "savings_cashflow")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

import sys, pickle, datetime
sys.path.append("../..")
from src.common.config import get_config
from src.common.glm_core import Evaluator
import pandas as pd
import numpy as np

cfg = get_config(catalog, schema)
VOLUME_PATH = f"/Volumes/{catalog}/{schema}/model_artifacts"

# COMMAND ----------
with open(f"{VOLUME_PATH}/latest_training_run.pkl", "rb") as f:
    artifact = pickle.load(f)

all_results = artifact["results"]

# COMMAND ----------
evaluator = Evaluator()

def compute_baseline_mape(train_df: pd.DataFrame) -> float:
    """Naive last-value forecast: predict balance = balance_lag_1."""
    pdf = train_df[train_df['balance'].notna() & train_df['balance_lag_1'].notna()].copy()
    if pdf.empty:
        return float("nan")
    pdf['naive_pred'] = pdf['balance_lag_1']
    try:
        return evaluator.mape(pdf, 'balance', 'naive_pred')
    except Exception:
        return float("nan")


rows = []
for product, res in all_results.items():
    metrics = res.get("metrics")
    if not metrics:
        print(f"WARNING: no metrics for product={product} — skipped")
        continue

    row = {"product": product, "evaluated_at": datetime.datetime.utcnow().isoformat()}
    row.update(metrics)

    # Baseline benchmark: naive last-value forecast
    train_df = res.get("train", pd.DataFrame())
    baseline = compute_baseline_mape(train_df)
    row["baseline_mape"] = round(baseline, 4) if not np.isnan(baseline) else None

    # Relative improvement over baseline (positive = GLM beats naive)
    glm_mape = metrics.get("mape_balance", float("nan"))
    if row["baseline_mape"] and not np.isnan(glm_mape):
        row["improvement_vs_baseline_pct"] = round(
            (baseline - glm_mape) / baseline * 100, 2
        )
        row["beats_baseline"] = bool(glm_mape < baseline)
    else:
        row["improvement_vs_baseline_pct"] = None
        row["beats_baseline"] = None

    rows.append(row)
    status = "✓" if row.get("beats_baseline") else "✗ WARN"
    print(f"  {status}  {product}: GLM={glm_mape:.2f}%  baseline={baseline:.2f}%  "
          f"improvement={row.get('improvement_vs_baseline_pct', 'N/A')}%")

metrics_df = pd.DataFrame(rows)
assert not metrics_df.empty, "No products produced evaluation metrics — aborting pipeline"

# Portfolio-level summary
portfolio_mape = metrics_df["mape_balance"].mean()
portfolio_rmse = metrics_df["rmse_balance"].mean()
products_beating_baseline = int(metrics_df["beats_baseline"].sum()) if "beats_baseline" in metrics_df else 0
baseline_coverage = products_beating_baseline / len(metrics_df) * 100

print(f"\nPortfolio MAPE (balance): {portfolio_mape:.2f}%")
print(f"Portfolio RMSE (balance): {portfolio_rmse:.2f}")
print(f"Products beating baseline: {products_beating_baseline}/{len(metrics_df)} ({baseline_coverage:.0f}%)")

# COMMAND ----------
spark.createDataFrame(metrics_df.astype(str)).write.mode("append").option(
    "mergeSchema", "true"
).saveAsTable(cfg.eval_metrics)

dbutils.jobs.taskValues.set(key="portfolio_mape",             value=float(portfolio_mape))
dbutils.jobs.taskValues.set(key="portfolio_rmse",             value=float(portfolio_rmse))
dbutils.jobs.taskValues.set(key="n_products_evaluated",       value=int(len(metrics_df)))
dbutils.jobs.taskValues.set(key="products_beating_baseline",  value=products_beating_baseline)
dbutils.jobs.taskValues.set(key="baseline_coverage_pct",      value=float(baseline_coverage))

print(f"Evaluation complete. {cfg.eval_metrics} updated with {len(metrics_df)} rows.")
