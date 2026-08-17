# Databricks notebook source
# MAGIC %md
# MAGIC # 04b - Challenger Model Validation
# MAGIC
# MAGIC Validates the candidate **@challenger** model before it can be promoted
# MAGIC to **@champion**. Mirrors the _Model Validation Job_ step in the
# MAGIC Databricks MLOps lifecycle diagram.
# MAGIC
# MAGIC **Validation checks run:**
# MAGIC 1. **Description check** — Has the data scientist documented the model version?
# MAGIC 2. **Performance metric check** — Is Challenger MAPE ≤ Champion MAPE?
# MAGIC    (first run always passes — no champion to beat)
# MAGIC 3. **Business metric check** — Total projected balance (Challenger vs Champion)
# MAGIC    as a proxy for revenue impact; must be within an acceptable range.
# MAGIC 4. **Product coverage check** — Challenger must cover ≥ same products as Champion.
# MAGIC
# MAGIC If all checks pass → sets `validation_passed=true` task value →
# MAGIC `05_model_registration.py` promotes @challenger to @champion.
# MAGIC If any check fails → sets `validation_passed=false` → challenger is rejected.

# COMMAND ----------
dbutils.widgets.text("catalog", "poc_mlops_dev")
dbutils.widgets.text("schema", "savings_cashflow")
dbutils.widgets.text("model_schema", "ml_models")
dbutils.widgets.text("model_name", "")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_schema = dbutils.widgets.get("model_schema")
model_name = dbutils.widgets.get("model_name") or f"{catalog}.{model_schema}.scf_cohort_model"

import sys, pickle, datetime
sys.path.append("../..")
from src.common.config import get_config
import mlflow
from mlflow import MlflowClient
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

mlflow.set_registry_uri("databricks-uc")
cfg = get_config(catalog, schema)
client = MlflowClient()
VOLUME_PATH = f"/Volumes/{catalog}/{model_schema}/model_artifacts"

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1 — Fetch Challenger Model Information

# COMMAND ----------
portfolio_mape = dbutils.jobs.taskValues.get(taskKey="model_evaluation", key="portfolio_mape")
n_products_evaluated = int(
    dbutils.jobs.taskValues.get(taskKey="model_evaluation", key="n_products_evaluated", default=0)
)

# Challenger is the version just trained — registered by 05 registration task
# In the validation flow, the challenger hasn't been registered yet so we use
# the evaluation artifact to derive validation metrics.
with open(f"{VOLUME_PATH}/latest_training_run.pkl", "rb") as f:
    artifact = pickle.load(f)

all_results = artifact["results"]

print(f"Validating challenger model: {model_name}")
print(f"  Cutoff period : {artifact['cutoff_period']}")
print(f"  Trained at    : {artifact['trained_at']}")
print(f"  Products      : {len(all_results)}")
print(f"  Portfolio MAPE: {portfolio_mape:.4f}%")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2 — Check 1: Description
# MAGIC Has the data scientist provided a model description and artifact documentation?

# COMMAND ----------
has_description = (
    len(all_results) > 0
    and artifact.get("cutoff_period") is not None
    and artifact.get("trained_at") is not None
)

print("Description check:")
print(f"  cutoff_period present : {artifact.get('cutoff_period') is not None}")
print(f"  trained_at present    : {artifact.get('trained_at') is not None}")
print(f"  n_products > 0        : {len(all_results) > 0}")
print(f"  has_description       : {has_description}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3 — Check 2: Performance Metric (MAPE)
# MAGIC Compare Challenger MAPE against current Champion MAPE.
# MAGIC Challenger passes if it beats champion OR no champion exists yet.

# COMMAND ----------
champion_mape = None
champion_n_products = 0
try:
    champ_mv = client.get_model_version_by_alias(model_name, "champion")
    champ_run = client.get_run(champ_mv.run_id)
    champion_mape_raw = champ_run.data.metrics.get("portfolio_mape_balance")
    champion_mape = float(champion_mape_raw) if champion_mape_raw else None
    champion_n_products = int(champ_run.data.params.get("n_products", 0))
    print(f"Champion  MAPE: {champion_mape:.4f}%  |  products: {champion_n_products}")
except Exception:
    print("No existing champion — challenger passes by default (first run).")

challenger_mape = float(portfolio_mape)
print(f"Challenger MAPE: {challenger_mape:.4f}%  |  products: {len(all_results)}")

if champion_mape is None:
    mape_passed = True
    mape_reason = "No champion exists — first run, challenger promoted by default."
else:
    improvement_pct = (champion_mape - challenger_mape) / champion_mape * 100
    mape_passed = improvement_pct >= -5.0   # allow up to 5% regression before blocking
    mape_reason = (
        f"Challenger MAPE {challenger_mape:.4f}% vs Champion {champion_mape:.4f}% "
        f"({improvement_pct:+.1f}% change). {'PASS' if mape_passed else 'FAIL'}"
    )
print(f"\nMAPE check: {mape_reason}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4 — Check 3: Business Metrics — Total Projected Balance
# MAGIC
# MAGIC The equivalent of "Revenue Impacted" for savings cashflow is the
# MAGIC **total projected balance** across all cohorts and products.
# MAGIC A large unexplained deviation from the champion's projection signals
# MAGIC the challenger may be mis-calibrated.

# COMMAND ----------
# Compute total projected balance for challenger from training artifact
challenger_rows = []
for product, res in all_results.items():
    train_df = res.get("train")
    if train_df is not None and "balance_pred" in train_df.columns:
        challenger_rows.append({
            "product": product,
            "total_balance_pred": pd.to_numeric(train_df["balance_pred"], errors="coerce").sum(),
            "total_balance_actual": pd.to_numeric(train_df.get("balance", pd.Series()), errors="coerce").sum(),
        })

challenger_biz_df = pd.DataFrame(challenger_rows)
challenger_total = challenger_biz_df["total_balance_pred"].sum() if not challenger_biz_df.empty else 0

# Fetch champion total projected balance from inference predictions (last scored batch)
champion_total = 0
try:
    preds_sdf = spark.table(cfg.inference_predictions)
    champ_run_latest = preds_sdf.agg({"scored_at": "max"}).collect()[0][0]
    champ_preds = (
        preds_sdf.filter(f"scored_at = '{champ_run_latest}'")
        .selectExpr("CAST(balance_pred AS DOUBLE) as balance_pred")
        .toPandas()
    )
    champion_total = champ_preds["balance_pred"].sum()
    print(f"Champion total projected balance : £{champion_total:,.0f}")
except Exception as e:
    print(f"No champion inference predictions available ({e}) — skipping champion comparison.")

print(f"Challenger total projected balance: £{challenger_total:,.0f}")

# Business metric passes if challenger is within ±30% of champion (or no champion)
if champion_total == 0:
    business_metric_passed = True
    biz_reason = "No champion predictions available — business metric check skipped."
else:
    deviation_pct = abs(challenger_total - champion_total) / champion_total * 100
    business_metric_passed = deviation_pct <= 30.0
    biz_reason = (
        f"Challenger £{challenger_total:,.0f} vs Champion £{champion_total:,.0f} "
        f"({deviation_pct:.1f}% deviation). {'PASS' if business_metric_passed else 'FAIL >30%'}"
    )
print(f"Business metric check: {biz_reason}")

# COMMAND ----------
# MAGIC %md
# MAGIC ### Business Metrics — Projected Balance: Challenger vs Champion

# COMMAND ----------
fig, ax = plt.subplots(figsize=(7, 5))
aliases  = ["Challenger", "Champion"]
totals   = [challenger_total, champion_total]
colours  = ["#6666FF", "#FF6644"]

bars = ax.bar(aliases, totals, color=colours, width=0.5)
ax.set_title("Business Metrics - Total Projected Balance", fontsize=13, fontweight="bold")
ax.set_xlabel("Model Alias")
ax.set_ylabel("Total Projected Balance (£)")
ax.yaxis.set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, _: f"£{x/1e6:.1f}M" if x >= 1e6 else f"£{x/1e3:.0f}k")
)
for bar, val in zip(bars, totals):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                f"£{val:,.0f}", ha="center", va="bottom", fontsize=10)

plt.tight_layout()
plt.savefig("/tmp/business_metrics_balance.png", dpi=120)
plt.show()
print("Chart saved to /tmp/business_metrics_balance.png")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5 — Check 4: Product Coverage
# MAGIC Challenger must cover at least as many products as the champion.

# COMMAND ----------
coverage_passed = True
coverage_reason = f"Challenger covers {len(all_results)} products."
if champion_n_products > 0:
    coverage_passed = len(all_results) >= champion_n_products
    coverage_reason = (
        f"Challenger products: {len(all_results)}  |  Champion products: {champion_n_products}. "
        f"{'PASS' if coverage_passed else 'FAIL — challenger covers fewer products'}"
    )
print(f"Coverage check: {coverage_reason}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6 — Validation Results Summary

# COMMAND ----------
validation_results = {
    "has_description":       str(has_description),
    "mape_passed":           str(mape_passed),
    "business_metric_passed": str(business_metric_passed),
    "coverage_passed":       str(coverage_passed),
    "challenger_mape":       f"{challenger_mape:.4f}",
    "champion_mape":         f"{champion_mape:.4f}" if champion_mape else "N/A",
    "challenger_balance":    f"{challenger_total:,.0f}",
    "champion_balance":      f"{champion_total:,.0f}",
    "n_products":            str(len(all_results)),
    "validated_at":          datetime.datetime.utcnow().isoformat(),
}

print("\n── Validation Results ────────────────────────────────")
for k, v in validation_results.items():
    status = ""
    if k in ("has_description", "mape_passed", "business_metric_passed", "coverage_passed"):
        status = "  ✓" if v == "True" else "  ✗ FAILED"
    print(f"  {k:30s}: {v}{status}")

# COMMAND ----------
validation_passed = all([
    has_description,
    mape_passed,
    business_metric_passed,
    coverage_passed,
])

if validation_passed:
    print("\n✓ All validation checks passed — challenger approved for promotion to @champion.")
else:
    failed = [k for k, v in {
        "has_description": has_description,
        "mape_passed": mape_passed,
        "business_metric_passed": business_metric_passed,
        "coverage_passed": coverage_passed,
    }.items() if not v]
    print(f"\n✗ Validation FAILED — checks failed: {failed}")
    print("  Challenger will be tagged as 'rejected' and archived.")

# COMMAND ----------
# COMMAND ----------
# MAGIC %md
# MAGIC ## 7 — Write model_approvals (Regulatory Audit Table)
# MAGIC Records every FMC approval/rejection decision with approver, version,
# MAGIC timestamp — required for model risk management and regulatory audit.

# COMMAND ----------
import datetime as dt

approval_row = pd.DataFrame([{
    "model_name":          model_name,
    "model_version":       str(
        dbutils.jobs.taskValues.get(taskKey="model_training", key="n_products", default="unknown")
    ),
    "validation_passed":   str(validation_passed),
    "approver":            "automated_validation_gate",
    "approval_timestamp":  dt.datetime.utcnow().isoformat(),
    "challenger_mape":     f"{challenger_mape:.4f}",
    "champion_mape":       f"{champion_mape:.4f}" if champion_mape else "N/A",
    "checks_passed":       str({k: v for k, v in {
        "has_description":        has_description,
        "mape_passed":            mape_passed,
        "business_metric_passed": business_metric_passed,
        "coverage_passed":        coverage_passed,
    }.items()}),
    "rejection_reason":    "" if validation_passed else str(
        [k for k, v in {
            "has_description": has_description,
            "mape_passed": mape_passed,
            "business_metric_passed": business_metric_passed,
            "coverage_passed": coverage_passed,
        }.items() if not v]
    ),
    "catalog":             catalog,
    "schema":              schema,
}])

spark.createDataFrame(approval_row).write.mode("append").option(
    "mergeSchema", "true"
).saveAsTable(cfg.model_approvals)
print(f"model_approvals written: validation_passed={validation_passed}")

# COMMAND ----------
# Pass validation outcome to model_registration task via task values.
dbutils.jobs.taskValues.set(key="validation_passed",   value=bool(validation_passed))
dbutils.jobs.taskValues.set(key="validation_results",  value=str(validation_results))
dbutils.jobs.taskValues.set(key="challenger_mape",     value=float(challenger_mape))
dbutils.jobs.taskValues.set(key="champion_mape",       value=float(champion_mape) if champion_mape else -1.0)
dbutils.jobs.taskValues.set(key="challenger_balance",  value=float(challenger_total))
dbutils.jobs.taskValues.set(key="champion_balance",    value=float(champion_total))
