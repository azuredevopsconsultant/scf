# Databricks notebook source
# MAGIC %md
# MAGIC # Unity Catalog — Create All Tables
# MAGIC
# MAGIC One-shot initialisation: creates all 16 Delta tables in the correct
# MAGIC catalog and schema with typed columns, table properties, and comments.
# MAGIC Safe to re-run — uses `CREATE TABLE IF NOT EXISTS`.

# COMMAND ----------
dbutils.widgets.text("catalog", "pd_dtl_ds_dev")
dbutils.widgets.text("schema",  "savings_cashflow")
catalog = dbutils.widgets.get("catalog")
schema  = dbutils.widgets.get("schema")

spark.sql(f"CREATE CATALOG  IF NOT EXISTS {catalog}")
spark.sql(f"CREATE SCHEMA   IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"CREATE VOLUME   IF NOT EXISTS {catalog}.{schema}.model_artifacts")

print(f"Initialising tables in {catalog}.{schema}")
created, skipped = [], []

def run(sql, name):
    try:
        spark.sql(sql)
        created.append(name)
        print(f"  ✓ {name}")
    except Exception as e:
        skipped.append(f"{name}: {e}")
        print(f"  - {name} (already exists or error: {e})")

# COMMAND ----------
# MAGIC %md ## BRONZE — Raw source tables

# COMMAND ----------
run(f"""
CREATE TABLE IF NOT EXISTS {catalog}.{schema}.raw_base_data (
  `account id`        STRING,
  `opening date`      DATE,
  `reporting period`  STRING,
  `product code`      STRING,
  `period balance`    DOUBLE,
  `£ withdrawal`      DOUBLE,
  `£ receipt`         DOUBLE,
  `£ internal transfer` DOUBLE,
  `interest rate`     DOUBLE,
  `account tenure (months)` DOUBLE
)
USING DELTA
TBLPROPERTIES (
  'delta.autoOptimize.optimizeWrite' = 'true',
  'delta.autoOptimize.autoCompact'   = 'true'
)
COMMENT 'Bronze: raw account-level savings data ingested from UC Volume parquet'
""", "raw_base_data")

run(f"""
CREATE TABLE IF NOT EXISTS {catalog}.{schema}.raw_moneyfacts_best_buy (
  period           DATE,
  mapping_category STRING,
  num_products     INT,
  best_buy         DOUBLE
)
USING DELTA
TBLPROPERTIES ('delta.autoOptimize.optimizeWrite' = 'true')
COMMENT 'Bronze: Moneyfacts best-buy rates per product category per month'
""", "raw_moneyfacts_best_buy")

run(f"""
CREATE TABLE IF NOT EXISTS {catalog}.{schema}.raw_qrm_products (
  `Product Code`   STRING,
  `QRM Category`   STRING
)
USING DELTA
TBLPROPERTIES ('delta.autoOptimize.optimizeWrite' = 'true')
COMMENT 'Bronze: QRM product category mapping'
""", "raw_qrm_products")

# COMMAND ----------
# MAGIC %md ## SILVER — Engineered features

# COMMAND ----------
run(f"""
CREATE TABLE IF NOT EXISTS {catalog}.{schema}.silver_agg_cohort (
  cohort               STRING,
  product              STRING,
  months_since_start   INT,
  reporting_period     STRING,
  balance              DOUBLE,
  withdrawals          DOUBLE,
  receipts             DOUBLE,
  transfers            DOUBLE,
  int_rate             DOUBLE,
  best_buy             DOUBLE,
  tenure_months        DOUBLE,
  `#accounts`          INT,
  balance_lag_1        DOUBLE,
  rec_prop             DOUBLE,
  outflow_prop         DOUBLE,
  withdrawal_prop_of_outflow DOUBLE,
  delta_to_best_buy    DOUBLE,
  dbb_range            STRING
)
USING DELTA
TBLPROPERTIES (
  'delta.autoOptimize.optimizeWrite' = 'true',
  'delta.autoOptimize.autoCompact'   = 'true'
)
COMMENT 'Silver: cohort-level aggregated features used for GLM training and inference seeding'
""", "silver_agg_cohort")

run(f"""
CREATE TABLE IF NOT EXISTS {catalog}.{schema}.feature_store_cohort (
  cohort               STRING  NOT NULL,
  product              STRING  NOT NULL,
  months_since_start   INT     NOT NULL,
  reporting_period     STRING,
  balance              DOUBLE,
  balance_lag_1        DOUBLE,
  int_rate             DOUBLE,
  best_buy             DOUBLE,
  delta_to_best_buy    DOUBLE,
  dbb_range            STRING,
  tenure_months        DOUBLE,
  `#accounts`          INT,
  rec_prop             DOUBLE,
  outflow_prop         DOUBLE,
  withdrawal_prop_of_outflow DOUBLE,
  CONSTRAINT pk_feature_store PRIMARY KEY (cohort, product, months_since_start)
)
USING DELTA
TBLPROPERTIES (
  'delta.autoOptimize.optimizeWrite' = 'true',
  'delta.enableChangeDataFeed'       = 'true'
)
COMMENT 'Silver: Databricks Feature Engineering Client table. Primary key: cohort × product × months_since_start'
""", "feature_store_cohort")

run(f"""
CREATE TABLE IF NOT EXISTS {catalog}.{schema}.inference_features (
  cohort               STRING  NOT NULL,
  product              STRING  NOT NULL,
  months_since_start   INT     NOT NULL,
  reporting_period     STRING,
  balance              DOUBLE,
  balance_lag_1        DOUBLE,
  int_rate             DOUBLE,
  best_buy             DOUBLE,
  delta_to_best_buy    DOUBLE,
  dbb_range            STRING,
  tenure_months        DOUBLE,
  `#accounts`          INT,
  CONSTRAINT pk_inference_features PRIMARY KEY (cohort, product, months_since_start)
)
USING DELTA
TBLPROPERTIES ('delta.autoOptimize.optimizeWrite' = 'true')
COMMENT 'Silver: inference-ready features for current scoring batch'
""", "inference_features")

# COMMAND ----------
# MAGIC %md ## GOLD — Model outputs

# COMMAND ----------
run(f"""
CREATE TABLE IF NOT EXISTS {catalog}.{schema}.inference_predictions (
  product              STRING,
  cohort               STRING,
  reporting_period     STRING,
  months_since_start   BIGINT,
  balance_pred         DOUBLE,
  receipts_pred        DOUBLE,
  withdrawals_pred     DOUBLE,
  transfers_pred       DOUBLE,
  balance              DOUBLE,
  receipts             DOUBLE,
  withdrawals          DOUBLE,
  transfers            DOUBLE,
  scored_at            STRING,
  model_name           STRING,
  model_version        STRING,
  model_alias          STRING
)
USING DELTA
TBLPROPERTIES (
  'delta.autoOptimize.optimizeWrite' = 'true',
  'delta.autoOptimize.autoCompact'   = 'true'
)
COMMENT 'Gold: GLM balance/flow predictions per cohort × product × period (append-only)'
""", "inference_predictions")

run(f"""
CREATE TABLE IF NOT EXISTS {catalog}.{schema}.model_eval_metrics (
  product                     STRING,
  evaluated_at                STRING,
  mape_balance                DOUBLE,
  rmse_balance                DOUBLE,
  mape_receipts               DOUBLE,
  rmse_receipts               DOUBLE,
  mape_withdrawals            DOUBLE,
  rmse_withdrawals            DOUBLE,
  baseline_mape               DOUBLE,
  improvement_vs_baseline_pct DOUBLE,
  beats_baseline              BOOLEAN
)
USING DELTA
TBLPROPERTIES ('delta.autoOptimize.optimizeWrite' = 'true')
COMMENT 'Gold: per-product MAPE/RMSE + baseline benchmark per training run (append-only)'
""", "model_eval_metrics")

# COMMAND ----------
# MAGIC %md ## MONITORING

# COMMAND ----------
run(f"""
CREATE TABLE IF NOT EXISTS {catalog}.{schema}.monitoring_metrics (
  product              STRING,
  run_timestamp        STRING,
  n_predictions        INT,
  null_prediction_rate DOUBLE,
  has_actuals          BOOLEAN,
  mape_balance         DOUBLE,
  rmse_balance         DOUBLE,
  logged_at            STRING
)
USING DELTA
TBLPROPERTIES ('delta.autoOptimize.optimizeWrite' = 'true')
COMMENT 'Monitoring: live MAPE/RMSE vs actuals + null prediction rate per inference run'
""", "monitoring_metrics")

run(f"""
CREATE TABLE IF NOT EXISTS {catalog}.{schema}.drift_results (
  feature         STRING,
  psi             DOUBLE,
  status          STRING,
  run_timestamp   STRING
)
USING DELTA
TBLPROPERTIES ('delta.autoOptimize.optimizeWrite' = 'true')
COMMENT 'Monitoring: PSI drift score per feature per inference run — OK / WARN / ALERT'
""", "drift_results")

run(f"""
CREATE TABLE IF NOT EXISTS {catalog}.{schema}.data_quality_results (
  check        STRING,
  passed       STRING,
  detail       STRING,
  validated_at STRING,
  pipeline     STRING
)
USING DELTA
TBLPROPERTIES ('delta.autoOptimize.optimizeWrite' = 'true')
COMMENT 'Monitoring: data quality check results per pipeline run'
""", "data_quality_results")

# COMMAND ----------
# MAGIC %md ## AUDIT — Regulatory tables

# COMMAND ----------
run(f"""
CREATE TABLE IF NOT EXISTS {catalog}.{schema}.dataset_versions (
  source_table    STRING,
  dataset_name    STRING,
  environment     STRING,
  delta_version   BIGINT,
  operation       STRING,
  recorded_at     STRING
)
USING DELTA
TBLPROPERTIES ('delta.autoOptimize.optimizeWrite' = 'true')
COMMENT 'Audit: Delta table version + timestamp per ingestion — enables VERSION AS OF exact-rerun'
""", "dataset_versions")

run(f"""
CREATE TABLE IF NOT EXISTS {catalog}.{schema}.training_feature_snapshots (
  cohort               STRING,
  product              STRING,
  months_since_start   STRING,
  reporting_period     STRING,
  balance              STRING,
  balance_lag_1        STRING,
  int_rate             STRING,
  best_buy             STRING,
  delta_to_best_buy    STRING,
  dbb_range            STRING,
  snapshot_timestamp   STRING,
  catalog              STRING,
  schema               STRING
)
USING DELTA
TBLPROPERTIES ('delta.autoOptimize.optimizeWrite' = 'true')
COMMENT 'Audit: frozen feature snapshot per training run — append-only, survives VACUUM'
""", "training_feature_snapshots")

run(f"""
CREATE TABLE IF NOT EXISTS {catalog}.{schema}.model_approvals (
  request_id         STRING,
  model_name         STRING,
  model_version      STRING,
  validation_passed  STRING,
  approver           STRING,
  approval_status    STRING,
  approval_timestamp STRING,
  challenger_mape    STRING,
  champion_mape      STRING,
  checks_passed      STRING,
  rejection_reason   STRING,
  catalog            STRING,
  schema             STRING
)
USING DELTA
TBLPROPERTIES ('delta.autoOptimize.optimizeWrite' = 'true')
COMMENT 'Audit: FMC approval/rejection decision, approver, version, timestamp — required for regulatory audit'
""", "model_approvals")

run(f"""
CREATE TABLE IF NOT EXISTS {catalog}.{schema}.model_cards (
  model_name         STRING,
  model_version      STRING,
  run_id             STRING,
  alias              STRING,
  cutoff_period      STRING,
  trained_at         STRING,
  n_products         STRING,
  portfolio_mape     STRING,
  features           STRING,
  labels             STRING,
  framework          STRING,
  promotion_decision STRING,
  promotion_reason   STRING,
  card_generated_at  STRING,
  catalog            STRING,
  schema             STRING
)
USING DELTA
TBLPROPERTIES ('delta.autoOptimize.optimizeWrite' = 'true')
COMMENT 'Audit: auto-generated model card per registered version — name, version, run_id, features, MAPE'
""", "model_cards")

run(f"""
CREATE TABLE IF NOT EXISTS {catalog}.{schema}.deployment_history (
  endpoint_name  STRING,
  action         STRING,
  model_name     STRING,
  traffic_map    STRING,
  deployed_at    STRING,
  catalog        STRING,
  schema         STRING
)
USING DELTA
TBLPROPERTIES ('delta.autoOptimize.optimizeWrite' = 'true')
COMMENT 'Audit: every endpoint create/update — version deployed, traffic split %, timestamp'
""", "deployment_history")

run(f"""
CREATE TABLE IF NOT EXISTS {catalog}.{schema}.rollback_events (
  endpoint_name       STRING,
  model_name          STRING,
  rollback_triggered  STRING,
  rollback_reason     STRING,
  reverted_to_version STRING,
  smoke_test_passed   STRING,
  serving_action      STRING,
  event_timestamp     STRING,
  catalog             STRING,
  schema              STRING
)
USING DELTA
TBLPROPERTIES ('delta.autoOptimize.optimizeWrite' = 'true')
COMMENT 'Audit: rollback reason, version reverted to, smoke test result — always written'
""", "rollback_events")

# COMMAND ----------
# MAGIC %md ## REFERENCE

# COMMAND ----------
run(f"""
CREATE TABLE IF NOT EXISTS {catalog}.{schema}.reference_feature_snapshot (
  cohort               STRING,
  product              STRING,
  months_since_start   STRING,
  reporting_period     STRING,
  int_rate             STRING,
  best_buy             STRING,
  delta_to_best_buy    STRING,
  tenure_months        STRING,
  `#accounts`          STRING
)
USING DELTA
TBLPROPERTIES ('delta.autoOptimize.optimizeWrite' = 'true')
COMMENT 'Reference: frozen training-time feature distribution used as PSI drift baseline'
""", "reference_feature_snapshot")

# COMMAND ----------
print(f"\n{'='*60}")
print(f"Unity Catalog initialisation complete: {catalog}.{schema}")
print(f"  Created : {len(created)}")
print(f"  Skipped : {len(skipped)}")
print("\nTables created:")
for t in created:
    print(f"  {catalog}.{schema}.{t}")
if skipped:
    print("\nSkipped (already exist):")
    for s in skipped:
        print(f"  {s}")
