# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Inferencing
# MAGIC Loads the model currently aliased `champion` (never `challenger` -
# MAGIC production traffic should only ever see a promoted model) and projects
# MAGIC balances/flows for the latest reporting periods.

# COMMAND ----------
dbutils.widgets.text("catalog", "pd_dtl_ds")
dbutils.widgets.text("schema", "savings_cashflow")
dbutils.widgets.text("model_name", "")
dbutils.widgets.text("model_alias", "champion")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name") or f"{catalog}.{schema}.scf_cohort_model"
model_alias = dbutils.widgets.get("model_alias") or "champion"

import sys
sys.path.append("../..")
from src.common.config import get_config
import mlflow
import pandas as pd

mlflow.set_registry_uri("databricks-uc")
cfg = get_config(catalog, schema)

# COMMAND ----------
model_uri = f"models:/{model_name}@{model_alias}"
model = mlflow.pyfunc.load_model(model_uri)
print(f"Loaded {model_uri}")

# COMMAND ----------
features_df = spark.table(cfg.inference_features).toPandas()
features_df['reporting_period'] = features_df['reporting_period'].astype(str)

predictions = model.predict(features_df)
assert not predictions.empty, "Model returned 0 predictions - check inference_features coverage"

# COMMAND ----------
predictions['scored_at'] = pd.Timestamp.utcnow().isoformat()
predictions['model_name'] = model_name
predictions['model_alias'] = model_alias

spark.createDataFrame(predictions.astype(str)).write.mode("append").option(
    "mergeSchema", "true"
).saveAsTable(cfg.inference_predictions)

dbutils.jobs.taskValues.set(key="n_predictions", value=int(len(predictions)))
print(f"Wrote {len(predictions)} predictions to {cfg.inference_predictions}")
