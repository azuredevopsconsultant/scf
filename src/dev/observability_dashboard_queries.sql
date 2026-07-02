-- Query 1: model quality over time
SELECT
  run_timestamp,
  product,
  mape_balance,
  rmse_balance,
  null_prediction_rate
FROM ${catalog}.${schema}.monitoring_metrics
ORDER BY run_timestamp DESC;

-- Query 2: drift summary
SELECT
  run_timestamp,
  feature,
  psi,
  status
FROM ${catalog}.${schema}.drift_results
ORDER BY run_timestamp DESC, psi DESC;

-- Query 3: data quality trend
SELECT
  check_name,
  check_status,
  checked_at,
  details
FROM ${catalog}.${schema}.data_quality_results
ORDER BY checked_at DESC;

-- Query 4: dataset lineage/version history
SELECT
  dataset_name,
  source_table,
  delta_version,
  operation,
  recorded_at,
  environment
FROM ${catalog}.${schema}.dataset_versions
ORDER BY recorded_at DESC;
