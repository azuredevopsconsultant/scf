# Databricks Enterprise MLOps Mapping

This file maps the requested enterprise checklist to concrete project assets.

1. Feature Store / Feature Engineering
- Feature engineering pipeline: src/training/02_data_preprocessing.py
- Feature store output table: src/common/config.py (feature_store_table)

2. Dataset Versioning
- Delta version capture helper: src/common/dataset_versioning.py
- Version tracking table: src/common/config.py (dataset_versions)
- Capture points: src/training/02_data_preprocessing.py and src/inference/02_data_preprocessing.py

3. Model Serving Endpoint
- Endpoint upsert script: src/dev/create_model_serving_endpoint.py
- Uses champion alias resolution from Unity Catalog model registry.

4. Lakehouse Monitoring
- Monitoring metrics pipeline: src/inference/04_model_monitoring.py
- Drift pipeline: src/inference/05_data_drift.py
- Monitoring tables configured in src/common/config.py

5. Retraining Trigger
- Trigger rules: src/common/retraining.py
- Runtime trigger task: src/inference/08_retraining_trigger.py
- Added to inference workflow in resources/inference_job.yml

6. Webhook Notifications
- Webhook sender: src/common/notifications.py (send_webhook_notification)
- Drift and confirmation integration: src/inference/05_data_drift.py, src/inference/07_confirmation_mail.py, src/training/06_confirmation_mail.py
- Bundle variable: databricks.yml (notification_webhook_url)

7. Unit and Integration Tests
- Existing unit tests: tests/test_glm_core.py
- Added trigger tests: tests/test_retraining.py

8. CI/CD Gates
- GitHub Actions pipeline: .github/workflows/ci.yml
- Gates: dependency install, pytest, databricks bundle validate -t dev

9. Observability Dashboards
- SQL starter pack for dashboard tiles and alerts: src/dev/observability_dashboard_queries.sql

## Deployment Notes

- Validate:
  databricks bundle validate -t dev
- Deploy:
  databricks bundle deploy -t dev --auto-approve
- If local terraform bootstrap fails, clear local state and redeploy:
  Remove-Item .databricks\bundle\dev -Recurse -Force
  databricks bundle deploy -t dev --auto-approve
