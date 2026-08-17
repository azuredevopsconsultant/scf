# Databricks notebook source
# MAGIC %md
# MAGIC # FMC Approval Notebook
# MAGIC
# MAGIC Run this notebook to **approve or reject** a pending challenger model.
# MAGIC
# MAGIC 1. Set `request_id` to the ID from the Teams/email notification
# MAGIC 2. Set `decision` to `APPROVED` or `REJECTED`
# MAGIC 3. Set `approver_name` to your name / employee ID
# MAGIC 4. Run all cells — the training pipeline will detect the decision and proceed

# COMMAND ----------
dbutils.widgets.text("catalog",       "poc_mlops_dev")
dbutils.widgets.text("schema",        "savings_cashflow")
dbutils.widgets.text("request_id",    "")
dbutils.widgets.text("decision",      "APPROVED")    # APPROVED or REJECTED
dbutils.widgets.text("approver_name", "")
dbutils.widgets.text("comments",      "")

catalog       = dbutils.widgets.get("catalog")
schema        = dbutils.widgets.get("schema")
request_id    = dbutils.widgets.get("request_id").strip().upper()
decision      = dbutils.widgets.get("decision").strip().upper()
approver_name = dbutils.widgets.get("approver_name").strip()
comments      = dbutils.widgets.get("comments").strip()

import sys, datetime
sys.path.append("../..")
from src.common.config import get_config
import pandas as pd

cfg = get_config(catalog, schema)

assert request_id,    "request_id is required"
assert approver_name, "approver_name is required"
assert decision in ("APPROVED", "REJECTED"), "decision must be APPROVED or REJECTED"

# COMMAND ----------
# Show current pending request for review
pending = spark.sql(f"""
    SELECT request_id, model_name, challenger_mape, champion_mape,
           validation_passed, approval_timestamp, approval_status
    FROM   {cfg.model_approvals}
    WHERE  request_id = '{request_id}'
    ORDER BY approval_timestamp DESC
    LIMIT  1
""").toPandas()

assert not pending.empty, f"No pending request found with request_id={request_id}"
display(pending)

# COMMAND ----------
# Write approval / rejection decision
decision_row = pd.DataFrame([{
    "request_id":         request_id,
    "model_name":         pending["model_name"].iloc[0],
    "model_version":      "pending_registration",
    "validation_passed":  pending["validation_passed"].iloc[0],
    "approver":           approver_name,
    "approval_status":    decision,
    "approval_timestamp": datetime.datetime.utcnow().isoformat(),
    "challenger_mape":    pending["challenger_mape"].iloc[0],
    "champion_mape":      pending["champion_mape"].iloc[0],
    "checks_passed":      pending["validation_passed"].iloc[0],
    "rejection_reason":   comments if decision == "REJECTED" else "",
    "catalog":            catalog,
    "schema":             schema,
}])

spark.createDataFrame(decision_row).write.mode("append").option(
    "mergeSchema", "true"
).saveAsTable(cfg.model_approvals)

icon = "✅" if decision == "APPROVED" else "❌"
print(f"{icon} {decision} recorded for request {request_id} by {approver_name}")
print("The training pipeline will detect this decision within 5 minutes.")
if comments:
    print(f"Comments: {comments}")
