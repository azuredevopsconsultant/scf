# Databricks notebook source
# MAGIC %md
# MAGIC # 08 - Retraining Trigger
# MAGIC Combines monitoring and drift signals into a single retraining recommendation.

# COMMAND ----------
dbutils.widgets.text("catalog", "pd_dtl_ds")
dbutils.widgets.text("schema", "savings_cashflow")
dbutils.widgets.text("notification_email", "")
dbutils.widgets.text("notification_webhook_url", "")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
notification_email = dbutils.widgets.get("notification_email")
notification_webhook_url = dbutils.widgets.get("notification_webhook_url")

import sys
sys.path.append("../..")
from src.common.config import get_config
from src.common.retraining import evaluate_retraining_need
from src.common.notifications import send_webhook_notification

cfg = get_config(catalog, schema)


def safe_task_value(task_key, key, default=None):
    try:
        return dbutils.jobs.taskValues.get(taskKey=task_key, key=key, default=default)
    except Exception:
        return default


drift_detected = bool(safe_task_value("data_drift", "drift_detected", False))
max_null_rate = float(safe_task_value("model_monitoring", "max_null_rate", 0.0) or 0.0)
portfolio_mape_raw = safe_task_value("model_monitoring", "portfolio_mape_monitoring", None)
portfolio_mape = None if portfolio_mape_raw is None else float(portfolio_mape_raw)

retrain_recommended, reason = evaluate_retraining_need(
    drift_detected=drift_detected,
    max_null_rate=max_null_rate,
    portfolio_mape=portfolio_mape,
    null_rate_threshold=cfg.RETRAIN_NULL_RATE_THRESHOLD,
    mape_threshold=cfg.RETRAIN_MAPE_THRESHOLD,
)

if retrain_recommended and notification_webhook_url:
    run_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
    send_webhook_notification(
        webhook_url=notification_webhook_url,
        title=f"[{catalog}] Retraining recommended",
        facts={
            "reason": reason,
            "drift_detected": str(drift_detected),
            "max_null_rate": f"{max_null_rate:.2%}",
            "portfolio_mape": "N/A" if portfolio_mape is None else f"{portfolio_mape:.2f}",
        },
        run_url=run_url,
    )

if retrain_recommended and notification_email:
    print(f"Retraining recommendation should be sent to {notification_email}")

print(reason)
dbutils.jobs.taskValues.set(key="retrain_recommended", value=bool(retrain_recommended))
dbutils.jobs.taskValues.set(key="retrain_reason", value=reason)
