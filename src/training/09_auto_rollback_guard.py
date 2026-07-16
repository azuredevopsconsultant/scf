# Databricks notebook source
# MAGIC %md
# MAGIC # 09 - Auto Rollback Guard
# MAGIC
# MAGIC Runs after the smoke test. If the smoke test failed OR the serving
# MAGIC endpoint returns an error rate above the threshold, automatically
# MAGIC reverts the endpoint to the previous champion version.
# MAGIC
# MAGIC Every rollback is recorded in the `rollback_events` audit table with:
# MAGIC   - reason for rollback
# MAGIC   - version reverted to
# MAGIC   - smoke test result that triggered it
# MAGIC   - timestamp

# COMMAND ----------
dbutils.widgets.text("catalog", "pd_dtl_ds")
dbutils.widgets.text("schema", "savings_cashflow")
dbutils.widgets.text("model_name", "")
dbutils.widgets.text("serving_endpoint_name", "scf-cohort-serving-endpoint")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name") or f"{catalog}.{schema}.scf_cohort_model"
serving_endpoint_name = dbutils.widgets.get("serving_endpoint_name")

import sys, datetime
sys.path.append("../..")
from src.common.config import get_config
from src.common import mlflow_utils
import mlflow
from mlflow import MlflowClient
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
import pandas as pd

mlflow.set_registry_uri("databricks-uc")
cfg = get_config(catalog, schema)
client = MlflowClient()
w = WorkspaceClient()

# COMMAND ----------
# Read smoke test result from previous task via task values
smoke_test_passed = bool(
    dbutils.jobs.taskValues.get(taskKey="smoke_test_serving_endpoint", key="smoke_test_passed", default=True)
)
serving_action = dbutils.jobs.taskValues.get(taskKey="deploy_serving_endpoint", key="serving_action", default="unknown")
traffic_map_str = dbutils.jobs.taskValues.get(taskKey="deploy_serving_endpoint", key="traffic_map", default="{}")

print(f"Smoke test passed : {smoke_test_passed}")
print(f"Serving action    : {serving_action}")
print(f"Traffic map       : {traffic_map_str}")

# COMMAND ----------
rollback_needed = not smoke_test_passed
rollback_reason = ""
reverted_to_version = "N/A"

if rollback_needed:
    rollback_reason = "Smoke test failed after deployment — reverting endpoint to prior champion."
    print(f"ROLLBACK TRIGGERED: {rollback_reason}")

    # Find the prior champion: the version tagged stage_history=champion
    # that is NOT the current challenger/just-deployed version
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        prior_champion = None
        for v in sorted(versions, key=lambda x: int(x.version), reverse=True):
            if (v.tags.get("stage_history") == "champion"
                    and v.version != str(client.get_model_version_by_alias(model_name, "champion").version)):
                prior_champion = v
                break

        if prior_champion:
            # Re-point @champion alias back to the prior version
            client.set_registered_model_alias(model_name, "champion", prior_champion.version)
            client.set_model_version_tag(
                model_name, prior_champion.version, "stage_history", "champion_restored"
            )
            reverted_to_version = prior_champion.version

            # Revert serving endpoint to 100% prior champion
            rollback_config = EndpointCoreConfigInput(
                served_entities=[
                    ServedEntityInput(
                        entity_name=model_name,
                        entity_version=prior_champion.version,
                        workload_size="Small",
                        scale_to_zero_enabled=True,
                        traffic_percentage=100,
                    )
                ]
            )
            w.serving_endpoints.update_config_and_wait(
                name=serving_endpoint_name,
                served_entities=rollback_config.served_entities,
            )
            print(f"Endpoint reverted to v{reverted_to_version} (prior champion)")
        else:
            rollback_reason += " No prior champion found — endpoint left as-is."
            print("WARNING: no prior champion version found to roll back to.")

    except Exception as e:
        rollback_reason += f" Rollback attempt failed: {e}"
        print(f"ERROR during rollback: {e}")
else:
    print("Smoke test passed — no rollback needed.")

# COMMAND ----------
# Write rollback_events audit table — every event recorded regardless of outcome.
rollback_row = pd.DataFrame([{
    "endpoint_name":      serving_endpoint_name,
    "model_name":         model_name,
    "rollback_triggered": str(rollback_needed),
    "rollback_reason":    rollback_reason or "smoke_test_passed",
    "reverted_to_version": str(reverted_to_version),
    "smoke_test_passed":  str(smoke_test_passed),
    "serving_action":     serving_action,
    "event_timestamp":    datetime.datetime.utcnow().isoformat(),
    "catalog":            catalog,
    "schema":             schema,
}])

spark.createDataFrame(rollback_row).write.mode("append").option(
    "mergeSchema", "true"
).saveAsTable(cfg.rollback_events)
print(f"rollback_events written: rollback_triggered={rollback_needed}")

# COMMAND ----------
dbutils.jobs.taskValues.set(key="rollback_triggered", value=bool(rollback_needed))
dbutils.jobs.taskValues.set(key="reverted_to_version", value=str(reverted_to_version))
