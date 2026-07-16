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

-- ─────────────────────────────────────────────────────────────────────────────
-- CHART QUERIES  (paste into Databricks SQL → use Line Chart visualisation)
-- ─────────────────────────────────────────────────────────────────────────────

-- Query 5: Balance — Actual vs Predicted per product (latest scored run)
-- Chart: Line chart | X = reporting_period | Series = metric | Y = value
SELECT
  reporting_period,
  product,
  SUM(CAST(balance      AS DOUBLE)) AS balance_actual,
  SUM(CAST(balance_pred AS DOUBLE)) AS balance_predicted
FROM ${catalog}.${schema}.inference_predictions
WHERE scored_at = (SELECT MAX(scored_at) FROM ${catalog}.${schema}.inference_predictions)
GROUP BY reporting_period, product
ORDER BY product, reporting_period;

-- Query 6: Receipts — Actual vs Predicted per product (latest scored run)
SELECT
  reporting_period,
  product,
  SUM(CAST(receipts      AS DOUBLE)) AS receipts_actual,
  SUM(CAST(receipts_pred AS DOUBLE)) AS receipts_predicted
FROM ${catalog}.${schema}.inference_predictions
WHERE scored_at = (SELECT MAX(scored_at) FROM ${catalog}.${schema}.inference_predictions)
GROUP BY reporting_period, product
ORDER BY product, reporting_period;

-- Query 7: Withdrawals — Actual vs Predicted per product (latest scored run)
SELECT
  reporting_period,
  product,
  SUM(CAST(withdrawals      AS DOUBLE)) AS withdrawals_actual,
  SUM(CAST(withdrawals_pred AS DOUBLE)) AS withdrawals_predicted
FROM ${catalog}.${schema}.inference_predictions
WHERE scored_at = (SELECT MAX(scored_at) FROM ${catalog}.${schema}.inference_predictions)
GROUP BY reporting_period, product
ORDER BY product, reporting_period;

-- Query 8: Transfers — Actual vs Predicted per product (latest scored run)
SELECT
  reporting_period,
  product,
  SUM(CAST(transfers      AS DOUBLE)) AS transfers_actual,
  SUM(CAST(transfers_pred AS DOUBLE)) AS transfers_predicted
FROM ${catalog}.${schema}.inference_predictions
WHERE scored_at = (SELECT MAX(scored_at) FROM ${catalog}.${schema}.inference_predictions)
GROUP BY reporting_period, product
ORDER BY product, reporting_period;

-- Query 9: Portfolio total — all products summed (Balance + Receipts)
-- Chart: Line chart | X = reporting_period | Y = portfolio_balance_actual / portfolio_balance_predicted
SELECT
  reporting_period,
  SUM(CAST(balance          AS DOUBLE)) AS portfolio_balance_actual,
  SUM(CAST(balance_pred     AS DOUBLE)) AS portfolio_balance_predicted,
  SUM(CAST(receipts         AS DOUBLE)) AS portfolio_receipts_actual,
  SUM(CAST(receipts_pred    AS DOUBLE)) AS portfolio_receipts_predicted,
  SUM(CAST(withdrawals      AS DOUBLE)) AS portfolio_withdrawals_actual,
  SUM(CAST(withdrawals_pred AS DOUBLE)) AS portfolio_withdrawals_predicted,
  SUM(CAST(transfers        AS DOUBLE)) AS portfolio_transfers_actual,
  SUM(CAST(transfers_pred   AS DOUBLE)) AS portfolio_transfers_predicted
FROM ${catalog}.${schema}.inference_predictions
WHERE scored_at = (SELECT MAX(scored_at) FROM ${catalog}.${schema}.inference_predictions)
GROUP BY reporting_period
ORDER BY reporting_period;

-- Query 10: MAPE trend over time per product (accuracy heatmap source)
-- Chart: Bar chart | X = product | Y = mape_balance | Color = run_timestamp
SELECT
  run_timestamp,
  product,
  CAST(mape_balance      AS DOUBLE) AS mape_balance,
  CAST(mape_receipts     AS DOUBLE) AS mape_receipts,
  CAST(mape_withdrawals  AS DOUBLE) AS mape_withdrawals,
  CAST(mape_transfers    AS DOUBLE) AS mape_transfers
FROM ${catalog}.${schema}.monitoring_metrics
WHERE has_actuals = 'True'
ORDER BY run_timestamp DESC, product;

-- Query 11: Ratio table — Receipt % by cohort month × rate bin (one product)
-- Replace 'Restricted NonISA' with target product
-- Chart: Heatmap | X = rate_bin | Y = cohort_month | Color = receipt_pct
SELECT
  cohort_month,
  rate_bin,
  receipt_pct
FROM ${catalog}.${schema}.training_ratio_tables
WHERE product = 'Restricted NonISA'
ORDER BY cohort_month, rate_bin;
