# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - Model Monitoring (Metrics)
# MAGIC Computes MAPE/RMSE on live inference with an actuals feedback loop:
# MAGIC   1. Latest predictions loaded from inference_predictions table.
# MAGIC   2. Actuals joined from silver_agg_cohort (the training Silver table)
# MAGIC      for any reporting_period where ground truth has since landed.
# MAGIC   3. Metrics computed per product and aggregated to portfolio MAPE.
# MAGIC   4. Results written to monitoring_metrics for BI dashboard / SQL alerts.
# MAGIC   5. Portfolio MAPE and null rate passed forward to retraining_trigger.

# COMMAND ----------
dbutils.widgets.text("catalog", "pd_dtl_ds")
dbutils.widgets.text("schema", "savings_cashflow")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

import sys, datetime
sys.path.append("../..")
from src.common.config import get_config
from src.common.glm_core import Evaluator
import pandas as pd

cfg = get_config(catalog, schema)

# COMMAND ----------
preds = spark.table(cfg.inference_predictions).toPandas()
latest_run = preds['scored_at'].max()
latest_preds = preds[preds['scored_at'] == latest_run].copy()

for col in ['balance', 'balance_pred', 'receipts', 'receipts_pred',
            'withdrawals', 'withdrawals_pred', 'transfers', 'transfers_pred']:
    latest_preds[col] = pd.to_numeric(latest_preds[col], errors='coerce')

# COMMAND ----------
# Actuals feedback loop: join ground-truth balances from silver_agg_cohort
# for any reporting_period that has since landed after this batch was scored.
# Predictions carry the predicted balance in balance_pred; we overwrite
# the balance column with the real actuals where available so MAPE is live.
try:
    actuals = spark.table(cfg.silver_agg_cohort).select(
        'cohort', 'product', 'reporting_period', 'balance'
    ).toPandas()
    actuals.columns = ['cohort', 'product', 'reporting_period', 'actual_balance']
    actuals['reporting_period'] = actuals['reporting_period'].astype(str)
    actuals['cohort'] = actuals['cohort'].astype(str)

    latest_preds['reporting_period'] = latest_preds['reporting_period'].astype(str)
    latest_preds['cohort'] = latest_preds['cohort'].astype(str)

    latest_preds = latest_preds.merge(
        actuals, on=['cohort', 'product', 'reporting_period'], how='left'
    )
    # Prefer joined actuals over any balance already in predictions
    mask = latest_preds['actual_balance'].notna()
    latest_preds.loc[mask, 'balance'] = latest_preds.loc[mask, 'actual_balance']
    latest_preds.drop(columns=['actual_balance'], inplace=True)
    n_joined = mask.sum()
    print(f"Actuals feedback loop: joined {n_joined} rows from silver_agg_cohort")
except Exception as e:
    print(f"WARNING: actuals join failed ({e}). Monitoring will use prediction-only metrics.")

# COMMAND ----------
evaluator = Evaluator()
rows = []
for product, grp in latest_preds.groupby('product'):
    actual_available = grp['balance'].notna().any()
    row = {
        "product": product,
        "run_timestamp": latest_run,
        "n_predictions": len(grp),
        "null_prediction_rate": grp['balance_pred'].isna().mean(),
        "has_actuals": bool(actual_available),
    }
    if actual_available:
        row["mape_balance"] = evaluator.mape(grp, 'balance', 'balance_pred')
        row["rmse_balance"] = evaluator.rmse(grp, 'balance', 'balance_pred')
    rows.append(row)

monitoring_df = pd.DataFrame(rows)
monitoring_df["logged_at"] = datetime.datetime.utcnow().isoformat()

# COMMAND ----------
spark.createDataFrame(monitoring_df.astype(str)).write.mode("append").option(
    "mergeSchema", "true"
).saveAsTable(cfg.monitoring_metrics)

# Operational guardrail: alert-worthy null-prediction spike fails the task
# outright (job-level on_failure email fires) rather than waiting for the
# weekly retrain to notice.
max_null_rate = monitoring_df["null_prediction_rate"].max()
portfolio_mape = None
if "mape_balance" in monitoring_df.columns:
    mape_series = pd.to_numeric(monitoring_df["mape_balance"], errors="coerce").dropna()
    if not mape_series.empty:
        portfolio_mape = float(mape_series.mean())

dbutils.jobs.taskValues.set(key="max_null_rate", value=float(max_null_rate))
if portfolio_mape is not None:
    dbutils.jobs.taskValues.set(key="portfolio_mape_monitoring", value=float(portfolio_mape))

assert max_null_rate < 0.5, (
    f"Null prediction rate {max_null_rate:.1%} exceeds 50% for at least one product - "
    "likely a broken model_alias load or feature mismatch."
)

print(f"Model monitoring complete: {len(monitoring_df)} products logged.")

 COMMAND ----------
# MAGIC %md
# MAGIC ## Actual vs Predicted Charts
# MAGIC One chart panel per product — Balance, Receipts, Withdrawals, Transfers.
# MAGIC Only rendered when actuals are available for the scored period.

# COMMAND ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings("ignore")

FLOW_PAIRS = [
    ("balance",     "balance_pred",     "Balance"),
    ("receipts",    "receipts_pred",    "Receipts"),
    ("withdrawals", "withdrawals_pred", "Withdrawals"),
    ("transfers",   "transfers_pred",   "Transfers"),
]

products_with_actuals = [
    p for p, g in latest_preds.groupby("product")
    if g["balance"].notna().any()
]

if not products_with_actuals:
    print("No actuals available yet for the scored period — charts will render after actuals land.")
else:
    for product in sorted(products_with_actuals):
        grp = (
            latest_preds[latest_preds["product"] == product]
            .assign(reporting_period=lambda d: d["reporting_period"].astype(str))
            .sort_values("reporting_period")
        )
        periods = grp["reporting_period"].tolist()
        x = range(len(periods))

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle(f"{product}  —  Actual vs Predicted  ({latest_run[:10]})", fontsize=13, fontweight="bold")

        for ax, (actual_col, pred_col, label) in zip(axes.flat, FLOW_PAIRS):
            actual = grp[actual_col].tolist()
            pred   = grp[pred_col].tolist()
            ax.plot(x, actual, marker="o", label="Actual",    color="#1f77b4", linewidth=2)
            ax.plot(x, pred,   marker="s", label="Predicted", color="#ff7f0e", linewidth=2, linestyle="--")
            ax.set_title(label, fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels(periods, rotation=30, ha="right", fontsize=8)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"£{v/1e6:.1f}M"))
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        display(fig)           # renders inline in the Databricks notebook
        plt.close(fig)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Portfolio-Level Actual vs Predicted (all products summed)

# COMMAND ----------
portfolio = (
    latest_preds
    .assign(reporting_period=lambda d: d["reporting_period"].astype(str))
    .groupby("reporting_period")[
        ["balance", "balance_pred", "receipts", "receipts_pred",
         "withdrawals", "withdrawals_pred", "transfers", "transfers_pred"]
    ].sum()
    .sort_index()
    .reset_index()
)

if portfolio["balance"].notna().any():
    periods = portfolio["reporting_period"].tolist()
    x = range(len(periods))

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"Portfolio Total  —  Actual vs Predicted  ({latest_run[:10]})", fontsize=13, fontweight="bold")

    for ax, (actual_col, pred_col, label) in zip(axes.flat, FLOW_PAIRS):
        ax.plot(x, portfolio[actual_col], marker="o", label="Actual",    color="#1f77b4", linewidth=2)
        ax.plot(x, portfolio[pred_col],   marker="s", label="Predicted", color="#ff7f0e", linewidth=2, linestyle="--")
        ax.set_title(label, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(periods, rotation=30, ha="right", fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"£{v/1e9:.2f}B"))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    display(fig)
    plt.close(fig)
else:
    print("No actuals available for portfolio chart.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## MAPE Heatmap — All Products × All Metrics

# COMMAND ----------
mape_cols = ["mape_balance", "mape_receipts", "mape_withdrawals", "mape_transfers"]
available_mape_cols = [c for c in mape_cols if c in monitoring_df.columns]

if available_mape_cols:
    heatmap_df = monitoring_df.set_index("product")[available_mape_cols].apply(
        pd.to_numeric, errors="coerce"
    )
    col_labels = [c.replace("mape_", "").capitalize() for c in available_mape_cols]

    fig, ax = plt.subplots(figsize=(len(available_mape_cols) * 2 + 2, len(heatmap_df) * 0.6 + 1.5))
    im = ax.imshow(heatmap_df.values, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=40)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(range(len(heatmap_df)))
    ax.set_yticklabels(heatmap_df.index, fontsize=9)

    for i in range(len(heatmap_df)):
        for j in range(len(available_mape_cols)):
            val = heatmap_df.values[i, j]
            if not pd.isna(val):
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=9,
                        color="black" if val < 20 else "white")

    plt.colorbar(im, ax=ax, label="MAPE %")
    ax.set_title(f"MAPE Heatmap by Product × Flow  ({latest_run[:10]})", fontsize=12, fontweight="bold")
    plt.tight_layout()
    display(fig)
    plt.close(fig)
else:
    print("MAPE columns not available — heatmap requires actuals.")
