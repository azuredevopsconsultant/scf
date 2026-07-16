# Databricks notebook source
# MAGIC %md
# MAGIC # RBAC Setup — Unity Catalog Permissions
# MAGIC
# MAGIC Run once after bundle deploy to enforce least-privilege access.
# MAGIC Requires CATALOG owner or metastore admin privileges.
# MAGIC
# MAGIC **Permission model:**
# MAGIC | Role | Principals | Access |
# MAGIC |---|---|---|
# MAGIC | ML Engineers | `ml-engineers` group | USE CATALOG, USE SCHEMA, SELECT on all tables, EXECUTE model |
# MAGIC | Data Analysts | `data-analysts` group | SELECT on inference/monitoring tables only |
# MAGIC | Data Scientists | `data-scientists` group | Full schema + model READ + WRITE feature store |
# MAGIC | Service Principals | spn-dev/preprod/prod | Full control in their catalog only |
# MAGIC | FMC Reviewers | `fmc-reviewers` group | SELECT on model_approvals, model_cards, model_eval_metrics |

# COMMAND ----------
dbutils.widgets.text("catalog", "pd_dtl_ds")
dbutils.widgets.text("schema",  "savings_cashflow")
dbutils.widgets.text("model_name", "")
catalog    = dbutils.widgets.get("catalog")
schema     = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name") or f"{catalog}.{schema}.scf_cohort_model"
full_schema = f"{catalog}.{schema}"

import sys
sys.path.append("../..")
from src.common.config import get_config
cfg = get_config(catalog, schema)

errors = []
def run_grant(sql):
    try:
        spark.sql(sql)
        print(f"  OK: {sql[:80]}...")
    except Exception as e:
        errors.append(f"FAILED: {sql[:80]}... → {e}")
        print(f"  SKIP (group may not exist yet): {e}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1 — Catalog and Schema USE grants

# COMMAND ----------
print("=== Catalog + Schema ===")
for group in ["ml-engineers", "data-analysts", "data-scientists", "fmc-reviewers"]:
    run_grant(f"GRANT USE CATALOG ON CATALOG {catalog} TO `{group}`")
    run_grant(f"GRANT USE SCHEMA ON SCHEMA {full_schema} TO `{group}`")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2 — Table SELECT grants

# COMMAND ----------
print("=== Table SELECT ===")
# All tables readable by ML engineers and data scientists
all_tables = [
    cfg.silver_agg_cohort, cfg.feature_store_table,
    cfg.inference_predictions, cfg.monitoring_metrics,
    cfg.drift_results, cfg.data_quality_results,
    cfg.model_eval_metrics, cfg.model_approvals, cfg.model_cards,
    cfg.deployment_history, cfg.rollback_events,
]
for tbl in all_tables:
    run_grant(f"GRANT SELECT ON TABLE {tbl} TO `ml-engineers`")
    run_grant(f"GRANT SELECT ON TABLE {tbl} TO `data-scientists`")

# Data analysts: inference outputs + monitoring only
analyst_tables = [cfg.inference_predictions, cfg.monitoring_metrics, cfg.drift_results]
for tbl in analyst_tables:
    run_grant(f"GRANT SELECT ON TABLE {tbl} TO `data-analysts`")

# FMC reviewers: audit tables only
fmc_tables = [cfg.model_approvals, cfg.model_cards, cfg.eval_metrics, cfg.model_eval_metrics]
for tbl in fmc_tables:
    run_grant(f"GRANT SELECT ON TABLE {tbl} TO `fmc-reviewers`")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3 — Model Registry grants

# COMMAND ----------
print("=== Model Registry ===")
run_grant(f"GRANT EXECUTE ON REGISTERED MODEL {model_name} TO `ml-engineers`")
run_grant(f"GRANT EXECUTE ON REGISTERED MODEL {model_name} TO `data-scientists`")
run_grant(f"GRANT SELECT ON REGISTERED MODEL {model_name} TO `fmc-reviewers`")
run_grant(f"GRANT SELECT ON REGISTERED MODEL {model_name} TO `data-analysts`")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4 — Feature Store MODIFY grant (data scientists write features)

# COMMAND ----------
print("=== Feature Store WRITE ===")
run_grant(f"GRANT MODIFY ON TABLE {cfg.feature_store_table} TO `data-scientists`")
run_grant(f"GRANT MODIFY ON TABLE {cfg.silver_agg_cohort} TO `data-scientists`")

# COMMAND ----------
print(f"\nRBAC setup complete for {full_schema}")
if errors:
    print(f"\n{len(errors)} grant(s) skipped (groups may not exist yet):")
    for e in errors:
        print(f"  {e}")
else:
    print("All grants applied successfully.")
