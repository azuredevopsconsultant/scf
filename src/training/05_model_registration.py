# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - Model Registration (Champion / Challenger)
# MAGIC
# MAGIC Implements the full Unity Catalog model lifecycle pattern:
# MAGIC
# MAGIC ```
# MAGIC  train run ──▶ log to MLflow ──▶ register @challenger ──▶ validate MAPE
# MAGIC                                                               │
# MAGIC                                            ┌──── Approved ───┘
# MAGIC                                            │              └── Rejected
# MAGIC                                            ▼                     ▼
# MAGIC                                        @champion           archived tag
# MAGIC ```
# MAGIC
# MAGIC Steps:
# MAGIC  1. Archive previous @challenger (if rejected last cycle)
# MAGIC  2. Log SCFCohortModel pyfunc + params/metrics to MLflow
# MAGIC  3. Register new version → always alias as @challenger
# MAGIC  4. Set @baseline on first-ever version
# MAGIC  5. Add rich description + metadata tags to version
# MAGIC  6. Evaluate: promote @challenger → @champion if MAPE improves > 15%
# MAGIC  7. Tag promoted version + update description accordingly

# COMMAND ----------
dbutils.widgets.text("catalog", "pd_dtl_ds")
dbutils.widgets.text("schema", "savings_cashflow")
dbutils.widgets.text("model_name", "")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name") or f"{catalog}.{schema}.scf_cohort_model"

import sys, pickle
sys.path.append("../..")
from src.common.config import get_config
from src.common.pyfunc_model import SCFCohortModel
from src.common import mlflow_utils

cfg = get_config(catalog, schema)
VOLUME_PATH = f"/Volumes/{catalog}/{schema}/model_artifacts"

# COMMAND ----------
with open(f"{VOLUME_PATH}/latest_training_run.pkl", "rb") as f:
    artifact = pickle.load(f)

all_results = artifact["results"]
portfolio_mape = dbutils.jobs.taskValues.get(taskKey="model_evaluation", key="portfolio_mape")
n_products_evaluated = dbutils.jobs.taskValues.get(
    taskKey="model_evaluation", key="n_products_evaluated", default=len(all_results)
)

# COMMAND ----------
# Step 1: Archive the previous @challenger if it was rejected last cycle.
# This implements the "Rejected → Archived" branch in the lifecycle diagram
# before we register the new candidate, so the @challenger alias is free.
mlflow_utils.archive_rejected_challenger(model_name)

# COMMAND ----------
# Step 2: Build SCFCohortModel pyfunc wrapping all product prediction tables.
product_tables = {
    product: {
        "rec_pct_tbl": res["rec_pct_tbl"],
        "outflow_pct_tbl": res["outflow_pct_tbl"],
        "transfer_pct_tbl": res["transfer_pct_tbl"],
    }
    for product, res in all_results.items()
}

scf_model = SCFCohortModel(
    product_tables=product_tables,
    trained_at=artifact["trained_at"],
    cutoff_period=artifact["cutoff_period"],
)

# Minimal valid input_example for signature inference
for product, res in all_results.items():
    input_example = res["train"].head(1)[
        ["product", "cohort", "reporting_period", "months_since_start", "dbb_range", "balance_lag_1"]
    ]
    break

# COMMAND ----------
# Step 3: Log to MLflow experiment with full params + metrics, then register.
import mlflow
from mlflow.models import infer_signature

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(f"/Shared/scf_cohort/{catalog}_training")

with mlflow.start_run(run_name=f"scf_cohort_training_{artifact['cutoff_period']}") as run:
    # Params
    mlflow.log_param("cutoff_period",   artifact["cutoff_period"])
    mlflow.log_param("trained_at",      artifact["trained_at"])
    mlflow.log_param("n_products",      len(all_results))
    mlflow.log_param("model_type",      "GLM_cohort_projection")
    mlflow.log_param("framework",       "statsmodels")
    # Metrics
    mlflow.log_metric("portfolio_mape_balance", portfolio_mape)
    mlflow.log_metric("n_products_evaluated",   n_products_evaluated)
    # Per-product MAPE
    for product, res in all_results.items():
        metrics = res.get("metrics") or {}
        if "mape_balance" in metrics:
            safe = product.replace(" ", "_").replace("(", "").replace(")", "")
            mlflow.log_metric(f"{safe}__mape_balance", metrics["mape_balance"])
    # Log model
    signature = infer_signature(input_example, scf_model.predict(None, input_example))
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=scf_model,
        signature=signature,
        input_example=input_example,
        pip_requirements=["statsmodels", "patsy", "pandas", "numpy"],
    )
    run_id = run.info.run_id

print(f"Logged MLflow run: {run_id}")

# COMMAND ----------
# Step 3 cont.: Register new version → always alias @challenger.
version = mlflow_utils.register_challenger(run_id, "model", model_name)
print(f"Registered {model_name} v{version} with alias '@challenger'")

# COMMAND ----------
# Step 4: Set @baseline alias on the very first registered version.
mlflow_utils.set_baseline_alias(model_name, version)

# COMMAND ----------
# Step 5: Set description and rich metadata tags on registered model + version.
# These appear in Unity Catalog Explorer → Catalog > model > version.
mlflow_utils.set_registered_model_description(
    model_name=model_name,
    description=(
        "SCF Cohort GLM model for savings balance and cashflow projection. "
        "Predicts monthly balance, receipts, withdrawals, and transfers per "
        "cohort x product combination using Generalised Linear Models. "
        "Used for customer retention strategy and regulatory stress-testing."
    ),
)

version_description = (
    f"Trained on cohorts up to {artifact['cutoff_period']}. "
    f"Portfolio MAPE (balance): {portfolio_mape:.2f}%. "
    f"Products covered: {len(all_results)}. "
    f"Trained at: {artifact['trained_at']}."
)
mlflow_utils.set_version_description(model_name, version, version_description)

mlflow_utils.set_rich_version_tags(
    model_name=model_name,
    version=version,
    trained_at=artifact["trained_at"],
    cutoff_period=artifact["cutoff_period"],
    n_products=len(all_results),
    portfolio_mape=portfolio_mape,
    catalog=catalog,
    schema=schema,
)
print(f"Description and tags set on v{version}")

# COMMAND ----------
# Step 6: Evaluate - promote @challenger → @champion if MAPE improves > threshold.
decision = mlflow_utils.evaluate_promotion(
    challenger_metrics={"mape": portfolio_mape},
    model_name=model_name,
    metric_key="mape",
    lower_is_better=True,
    improvement_threshold_pct=cfg.MAPE_PROMOTION_THRESHOLD,
)
print(decision["reason"])

if decision["promoted"]:
    # Step 7a: Promote → update tags + description to reflect @champion status
    promoted_version = mlflow_utils.promote_challenger_to_champion(model_name)
    mlflow_utils.set_version_description(
        model_name, promoted_version,
        f"[CHAMPION] {version_description}"
    )
    from mlflow import MlflowClient
    MlflowClient().set_model_version_tag(
        model_name, str(promoted_version), "stage_history", "champion"
    )
    print(f"Promoted {model_name} v{promoted_version} to '@champion'")
else:
    # Step 7b: Rejected - tag with rejection reason (will be archived next cycle)
    from mlflow import MlflowClient
    MlflowClient().set_model_version_tag(
        model_name, str(version), "stage_history", "rejected"
    )
    MlflowClient().set_model_version_tag(
        model_name, str(version), "rejection_reason",
        f"MAPE {portfolio_mape:.2f}% did not improve champion by {cfg.MAPE_PROMOTION_THRESHOLD}%"
    )
    print(f"Challenger v{version} rejected — champion unchanged.")

# COMMAND ----------
dbutils.jobs.taskValues.set(key="registered_version", value=str(version))
dbutils.jobs.taskValues.set(key="promoted",           value=bool(decision["promoted"]))
dbutils.jobs.taskValues.set(key="promotion_reason",   value=decision["reason"])
dbutils.jobs.taskValues.set(key="model_name",         value=model_name)

# COMMAND ----------
# Write model_cards audit table — auto-generated card per registered version.
# Provides a human-readable record: model name, version, run ID, feature list,
# target labels, portfolio MAPE. Visible in Unity Catalog governance layer.
import datetime as dt

feature_list = [
    "product", "cohort", "reporting_period", "months_since_start",
    "dbb_range", "balance_lag_1", "int_rate", "best_buy",
    "delta_to_best_buy", "tenure_months", "#accounts",
]
label_list = ["balance_pred", "receipts_pred", "withdrawals_pred", "transfers_pred"]

card_row = pd.DataFrame([{
    "model_name":        model_name,
    "model_version":     str(version),
    "run_id":            run_id,
    "alias":             "champion" if decision["promoted"] else "challenger",
    "cutoff_period":     artifact["cutoff_period"],
    "trained_at":        artifact["trained_at"],
    "n_products":        str(len(all_results)),
    "portfolio_mape":    f"{portfolio_mape:.4f}",
    "features":          str(feature_list),
    "labels":            str(label_list),
    "framework":         "statsmodels GLM",
    "promotion_decision": str(decision["promoted"]),
    "promotion_reason":  decision["reason"],
    "card_generated_at": dt.datetime.utcnow().isoformat(),
    "catalog":           catalog,
    "schema":            schema,
}])

spark.createDataFrame(card_row).write.mode("append").option(
    "mergeSchema", "true"
).saveAsTable(cfg.model_cards)
print(f"model_cards written for version {version}")

print(f"\nModel lifecycle summary:")
print(f"  Registered version : {version}")
print(f"  Alias assigned     : @challenger")
print(f"  Promoted to @champion: {decision['promoted']}")
print(f"  Reason             : {decision['reason']}")
print(f"  UC path            : {model_name}")
