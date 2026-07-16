# Databricks notebook source
# MAGIC %md
# MAGIC # 05a - FMC Second-Line Validation Gate
# MAGIC
# MAGIC **Human-in-the-loop** step required before any @challenger can be
# MAGIC promoted to @champion. Implements the _Second-Line Validation (FMC gate)_
# MAGIC shown in the MLOps Architecture diagram.
# MAGIC
# MAGIC **Flow:**
# MAGIC ```
# MAGIC 04b_model_validation  ──▶  05a_fmc_validation_gate
# MAGIC                                │
# MAGIC                     Write PENDING entry to model_approvals
# MAGIC                     Send notification to FMC reviewers
# MAGIC                                │
# MAGIC                     Poll model_approvals every 5 min
# MAGIC                     for up to fmc_timeout_minutes (default 1440 = 24h)
# MAGIC                                │
# MAGIC                    ┌──────────┴──────────┐
# MAGIC                 APPROVED             REJECTED / TIMEOUT
# MAGIC                    │                      │
# MAGIC            passes task            fails task → champion unchanged
# MAGIC ```
# MAGIC
# MAGIC **FMC reviewer action:**
# MAGIC Run `src/training/fmc_approve.py` with `decision=APPROVED` or `REJECTED`.

# COMMAND ----------
dbutils.widgets.text("catalog", "pd_dtl_ds")
dbutils.widgets.text("schema", "savings_cashflow")
dbutils.widgets.text("model_name", "")
dbutils.widgets.text("fmc_timeout_minutes", "1440")   # 24h default
dbutils.widgets.text("notification_email", "")
dbutils.widgets.text("notification_webhook_url", "")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name") or f"{catalog}.{schema}.scf_cohort_model"
fmc_timeout_minutes = int(dbutils.widgets.get("fmc_timeout_minutes") or 1440)
notification_email = dbutils.widgets.get("notification_email")
notification_webhook_url = dbutils.widgets.get("notification_webhook_url")

import sys, datetime, time, uuid
sys.path.append("../..")
from src.common.config import get_config
from src.common.notifications import send_webhook_notification
import pandas as pd

cfg = get_config(catalog, schema)
POLL_INTERVAL_SECONDS = 300   # check every 5 minutes

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1 — Retrieve challenger metrics from upstream task values

# COMMAND ----------
challenger_mape   = float(dbutils.jobs.taskValues.get(
    taskKey="model_validation", key="challenger_mape", default=-1))
champion_mape     = float(dbutils.jobs.taskValues.get(
    taskKey="model_validation", key="champion_mape",   default=-1))
challenger_balance = float(dbutils.jobs.taskValues.get(
    taskKey="model_validation", key="challenger_balance", default=0))
auto_validation_passed = bool(dbutils.jobs.taskValues.get(
    taskKey="model_validation", key="validation_passed", default=False))

# Generate a unique approval request ID so FMC can reference it
request_id = str(uuid.uuid4())[:8].upper()

print(f"FMC Review Request ID : {request_id}")
print(f"Challenger MAPE       : {challenger_mape:.4f}%")
print(f"Champion MAPE         : {champion_mape:.4f}%")
print(f"Challenger Balance    : £{challenger_balance:,.0f}")
print(f"Auto-validation passed: {auto_validation_passed}")
print(f"FMC timeout           : {fmc_timeout_minutes} minutes")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2 — Write PENDING entry to model_approvals

# COMMAND ----------
pending_row = pd.DataFrame([{
    "request_id":          request_id,
    "model_name":          model_name,
    "model_version":       "pending_registration",
    "validation_passed":   str(auto_validation_passed),
    "approver":            "PENDING_FMC_REVIEW",
    "approval_status":     "PENDING",
    "approval_timestamp":  datetime.datetime.utcnow().isoformat(),
    "challenger_mape":     f"{challenger_mape:.4f}",
    "champion_mape":       f"{champion_mape:.4f}" if champion_mape > 0 else "N/A",
    "checks_passed":       str(auto_validation_passed),
    "rejection_reason":    "",
    "catalog":             catalog,
    "schema":              schema,
}])

spark.createDataFrame(pending_row).write.mode("append").option(
    "mergeSchema", "true"
).saveAsTable(cfg.model_approvals)

print(f"Written PENDING FMC review request {request_id} to {cfg.model_approvals}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3 — Notify FMC reviewers

# COMMAND ----------
run_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()

if notification_webhook_url:
    send_webhook_notification(
        webhook_url=notification_webhook_url,
        title=f"[{catalog}] FMC Second-Line Validation Required — {request_id}",
        facts={
            "request_id":          request_id,
            "challenger_mape":     f"{challenger_mape:.4f}%",
            "champion_mape":       f"{champion_mape:.4f}%" if champion_mape > 0 else "N/A",
            "auto_validation":     str(auto_validation_passed),
            "action_required":     "Run fmc_approve.py with APPROVED or REJECTED",
            "timeout":             f"{fmc_timeout_minutes} minutes",
        },
        run_url=run_url,
    )

print(f"FMC notification sent. Reviewer must run fmc_approve.py with request_id={request_id}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4 — Poll model_approvals for FMC decision

# COMMAND ----------
deadline = datetime.datetime.utcnow() + datetime.timedelta(minutes=fmc_timeout_minutes)
fmc_decision = "PENDING"
approved_by  = ""

while datetime.datetime.utcnow() < deadline:
    result = spark.sql(f"""
        SELECT approval_status, approver
        FROM   {cfg.model_approvals}
        WHERE  request_id = '{request_id}'
          AND  approval_status IN ('APPROVED', 'REJECTED')
        ORDER BY approval_timestamp DESC
        LIMIT  1
    """).collect()

    if result:
        fmc_decision = result[0]["approval_status"]
        approved_by  = result[0]["approver"]
        print(f"FMC decision received: {fmc_decision} by {approved_by}")
        break

    remaining = (deadline - datetime.datetime.utcnow()).seconds // 60
    print(f"[{datetime.datetime.utcnow().strftime('%H:%M:%S')}] Still PENDING — "
          f"{remaining} min remaining. Next check in {POLL_INTERVAL_SECONDS//60} min.")
    time.sleep(POLL_INTERVAL_SECONDS)
else:
    fmc_decision = "TIMEOUT"
    print(f"FMC timeout after {fmc_timeout_minutes} minutes — treating as REJECTED.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5 — Set task values and assert decision

# COMMAND ----------
fmc_approved = fmc_decision == "APPROVED"

dbutils.jobs.taskValues.set(key="fmc_approved",  value=bool(fmc_approved))
dbutils.jobs.taskValues.set(key="fmc_decision",  value=fmc_decision)
dbutils.jobs.taskValues.set(key="fmc_approver",  value=approved_by)
dbutils.jobs.taskValues.set(key="request_id",    value=request_id)

print(f"\nFMC Gate result: {fmc_decision}")
assert fmc_approved, (
    f"FMC second-line validation {fmc_decision} (request_id={request_id}). "
    "Challenger will not be promoted. Re-trigger training job after FMC sign-off."
)
print("FMC APPROVED — proceeding to model registration.")
