# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - Model Registration (Champion / Challenger)
# MAGIC Packages every product's prediction tables into one `SCFCohortModel`
# MAGIC pyfunc, logs + registers it in the Unity Catalog Model Registry, always
# MAGIC aliases it `challenger`, and promotes to `champion` only if it beats the
# MAGIC current champion's portfolio MAPE by the configured threshold.

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

# COMMAND ----------
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

# Minimal valid input_example for signature inference - one row per product
input_example = None
for product, res in all_results.items():
    train_sample = res["train"].head(1)[
        ["product", "cohort", "reporting_period", "months_since_start", "dbb_range", "balance_lag_1"]
    ]
    input_example = train_sample if input_example is None else input_example
    break

# COMMAND ----------
import mlflow
from mlflow.models import infer_signature

mlflow.set_registry_uri("databricks-uc")

with mlflow.start_run(run_name="scf_cohort_training") as run:
    mlflow.log_param("cutoff_period", artifact["cutoff_period"])
    mlflow.log_param("n_products", len(all_results))
    mlflow.log_metric("portfolio_mape_balance", portfolio_mape)

    signature = infer_signature(input_example, scf_model.predict(None, input_example))
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=scf_model,
        signature=signature,
        input_example=input_example,
        pip_requirements=["statsmodels", "patsy", "pandas", "numpy"],
    )
    run_id = run.info.run_id

version = mlflow_utils.register_challenger(run_id, "model", model_name)
print(f"Registered {model_name} v{version} with alias 'challenger'")

# COMMAND ----------
decision = mlflow_utils.evaluate_promotion(
    challenger_metrics={"mape": portfolio_mape},
    model_name=model_name,
    metric_key="mape",
    lower_is_better=True,
    improvement_threshold_pct=cfg.MAPE_PROMOTION_THRESHOLD,
)
print(decision["reason"])

if decision["promoted"]:
    promoted_version = mlflow_utils.promote_challenger_to_champion(model_name)
    print(f"Promoted {model_name} v{promoted_version} to 'champion'")
else:
    print("Challenger did NOT beat current champion - champion unchanged. "
          "Review portfolio_mape trend in the eval_metrics table before next retrain.")

# COMMAND ----------
dbutils.jobs.taskValues.set(key="registered_version", value=str(version))
dbutils.jobs.taskValues.set(key="promoted", value=bool(decision["promoted"]))
dbutils.jobs.taskValues.set(key="promotion_reason", value=decision["reason"])
