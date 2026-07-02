# Databricks notebook source
# MAGIC %md
# MAGIC # 07 - Deploy/Update Serving Endpoint
# MAGIC Deploys the current champion (or challenger fallback) to the configured
# MAGIC Databricks Model Serving endpoint.

# COMMAND ----------
dbutils.widgets.text("catalog", "pd_dtl_ds")
dbutils.widgets.text("schema", "savings_cashflow")
dbutils.widgets.text("model_name", "")
dbutils.widgets.text("serving_endpoint_name", "scf-cohort-serving-endpoint")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name") or f"{catalog}.{schema}.scf_cohort_model"
serving_endpoint_name = dbutils.widgets.get("serving_endpoint_name")

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

mlflow.set_registry_uri("databricks-uc")


def resolve_served_version(full_model_name: str) -> tuple[str, str]:
    client = mlflow.MlflowClient(registry_uri="databricks-uc")
    for alias in ["champion", "challenger"]:
        try:
            version = client.get_model_version_by_alias(full_model_name, alias).version
            return str(version), alias
        except Exception:
            pass
    raise RuntimeError(
        f"No champion/challenger alias found for {full_model_name}. "
        "Ensure model_registration ran successfully."
    )


served_version, alias_used = resolve_served_version(model_name)
w = WorkspaceClient()
config = EndpointCoreConfigInput(
    served_entities=[
        ServedEntityInput(
            entity_name=model_name,
            entity_version=served_version,
            workload_size="Small",
            scale_to_zero_enabled=True,
        )
    ]
)

action = "updated"
try:
    w.serving_endpoints.get(serving_endpoint_name)
    w.serving_endpoints.update_config_and_wait(
        name=serving_endpoint_name,
        served_entities=config.served_entities,
    )
except Exception:
    action = "created"
    w.serving_endpoints.create_and_wait(name=serving_endpoint_name, config=config)

print(
    f"Serving endpoint {serving_endpoint_name} {action}: "
    f"{model_name} v{served_version} ({alias_used})"
)

dbutils.jobs.taskValues.set(key="serving_endpoint_name", value=serving_endpoint_name)
dbutils.jobs.taskValues.set(key="serving_model_version", value=served_version)
dbutils.jobs.taskValues.set(key="serving_alias_used", value=alias_used)
dbutils.jobs.taskValues.set(key="serving_action", value=action)
