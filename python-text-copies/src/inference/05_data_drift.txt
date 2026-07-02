# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - Data Drift (Alert + Mail on Drift)
# MAGIC Compares the current scoring batch's feature distributions against a
# MAGIC frozen training-time reference snapshot using PSI
# MAGIC (`src/common/drift_checks.py`). Sends an email **only when drift is
# MAGIC actually detected** - see `resources/inference_job.yml` for why this is
# MAGIC a custom send rather than a blanket job-level notification.

# COMMAND ----------
dbutils.widgets.text("catalog", "pd_dtl_ds")
dbutils.widgets.text("schema", "savings_cashflow")
dbutils.widgets.text("notification_email", "")
dbutils.widgets.text("notification_webhook_url", "")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
notification_email = dbutils.widgets.get("notification_email")
notification_webhook_url = dbutils.widgets.get("notification_webhook_url")

import sys, datetime
sys.path.append("../..")
from src.common.config import get_config
from src.common.drift_checks import compute_drift_report, any_alerts
from src.common.notifications import send_drift_alert, send_webhook_notification

cfg = get_config(catalog, schema)

DRIFT_COLUMNS = ['int_rate', 'best_buy', 'delta_to_best_buy', 'tenure_months', '#accounts']

# COMMAND ----------
current_pdf = spark.table(cfg.inference_features).toPandas()

# Reference snapshot: frozen once at first training run, refreshed
# deliberately (not automatically) after a validated retrain, so the drift
# baseline doesn't silently chase the incoming data.
try:
    reference_pdf = spark.table(cfg.volume_reference_data).toPandas()
except Exception:
    print("No reference snapshot found yet - seeding it from the current batch. "
          "Drift will only be meaningful from the next run onward.")
    spark.createDataFrame(current_pdf.astype(str)).write.mode("overwrite").saveAsTable(
        cfg.volume_reference_data
    )
    reference_pdf = current_pdf

# COMMAND ----------
for col in DRIFT_COLUMNS:
    current_pdf[col] = current_pdf[col].astype(float)
    reference_pdf[col] = reference_pdf[col].astype(float)

drift_report = compute_drift_report(
    reference_pdf, current_pdf, DRIFT_COLUMNS,
    warn_threshold=cfg.PSI_WARN_THRESHOLD, alert_threshold=cfg.PSI_ALERT_THRESHOLD,
)
print(drift_report)

drift_report["run_timestamp"] = datetime.datetime.utcnow().isoformat()
spark.createDataFrame(drift_report.astype(str)).write.mode("append").option(
    "mergeSchema", "true"
).saveAsTable(cfg.drift_results)

# COMMAND ----------
drift_detected = any_alerts(drift_report)
dbutils.jobs.taskValues.set(key="drift_detected", value=bool(drift_detected))

if drift_detected and notification_email:
    run_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
    alerted_features = drift_report[drift_report["status"] == "ALERT"]
    send_drift_alert(
        to_addresses=[notification_email],
        environment=dbutils.widgets.get("catalog"),
        drift_summary=dict(zip(alerted_features["feature"], alerted_features["psi"].round(3))),
        run_url=run_url,
    )
    if notification_webhook_url:
        send_webhook_notification(
            webhook_url=notification_webhook_url,
            title=f"[{catalog}] Data drift detected",
            facts={
                "alerted_features": ", ".join(alerted_features["feature"].astype(str).tolist()),
                "max_psi": str(alerted_features["psi"].max().round(3)),
            },
            run_url=run_url,
        )
    print(f"Drift alert email sent for features: {list(alerted_features['feature'])}")
elif drift_detected:
    print("Drift detected but no notification_email configured - skipping mail.")
else:
    print("No significant drift detected.")
