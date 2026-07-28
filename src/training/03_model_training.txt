# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Model Training
# MAGIC Runs `ModelingPipeline.run_for_product()` for every product with:
# MAGIC
# MAGIC **1. Hyperparameter grid search** — `month_end` × `drop_month2`
# MAGIC    Each trial logged as a **nested MLflow child run** (visible in MLflow
# MAGIC    UI as a collapsible group under the parent HPO run).
# MAGIC
# MAGIC **2. Baseline benchmark** — naive last-value forecast (balance_lag_1 as
# MAGIC    prediction). GLM must beat baseline MAPE or a warning is raised.
# MAGIC
# MAGIC **3. Rolling-origin cross-validation** — 3 temporal train/test splits
# MAGIC    to estimate generalisation error before the final fit on full data.

# COMMAND ----------
dbutils.widgets.text("catalog", "pd_dtl_ds")
dbutils.widgets.text("schema", "savings_cashflow")
dbutils.widgets.text("model_schema", "ml_models")
dbutils.widgets.text("cutoff_period", "2025-01")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_schema = dbutils.widgets.get("model_schema")
cutoff_period = dbutils.widgets.get("cutoff_period")

import sys, pickle, datetime, mlflow
sys.path.append("../..")
from src.common.config import get_config
from src.common.glm_core import ModelingPipeline

cfg = get_config(catalog, schema)

# COMMAND ----------
agg_df = spark.table(cfg.silver_agg_cohort).toPandas()
agg_df['reporting_period'] = agg_df['reporting_period'].astype('period[M]')
agg_df['cohort'] = agg_df['cohort'].astype('period[M]')

# COMMAND ----------
# Hyperparameter grid: search over month_end horizon and drop_month2 flag.
# Best config per product is selected by lowest balance MAPE on the
# validation window (periods after cutoff_period).
PARAM_GRID = [
    {"month_end": 36, "drop_month2": True},
    {"month_end": 48, "drop_month2": True},
    {"month_end": 50, "drop_month2": True},
    {"month_end": 50, "drop_month2": False},
    {"month_end": 60, "drop_month2": True},
]

# Rolling-origin CV splits: train up to each cutoff, test on next window.
# Uses 3 folds — enough to estimate generalisation error for quarterly retrains.
from src.common.glm_core import Evaluator
import numpy as np

def rolling_origin_cv(pipeline, product, df, cutoffs, month_end, drop_month2):
    """3-fold temporal CV. Returns mean MAPE across folds."""
    fold_mapes = []
    for cutoff in cutoffs:
        try:
            res = pipeline.run_for_product(
                PRODUCT=product,
                cutoff_period=cutoff,
                drop_month2=drop_month2,
                month_start=2, month_end=month_end,
                proj_start_period=cutoff,
                proj_end_period=str(df['reporting_period'].max()),
                proj_cohort_cutoff='2024-12',
                save_csv=False,
            )
            m = (res.get("metrics") or {}).get("mape_balance", float("inf"))
            if m < float("inf"):
                fold_mapes.append(m)
        except Exception:
            pass
    return float(np.mean(fold_mapes)) if fold_mapes else float("inf")


def baseline_mape(df, product):
    """Naive last-value benchmark: predict balance = balance_lag_1."""
    evaluator = Evaluator()
    pdf = df[df['product'] == product].copy()
    pdf = pdf[pdf['balance'].notna() & pdf['balance_lag_1'].notna()]
    if pdf.empty:
        return float("inf")
    pdf['naive_pred'] = pdf['balance_lag_1']
    return evaluator.mape(pdf, 'balance', 'naive_pred')


# Derive 3 rolling-origin cutoffs spaced ~6 months before the main cutoff
from pandas import Period
main_cutoff = Period(cutoff_period, freq='M')
cv_cutoffs = [
    str(main_cutoff - 12),
    str(main_cutoff - 6),
    str(main_cutoff - 3),
]

pipeline = ModelingPipeline(agg_df)
all_results = {}
best_params_per_product = {}
benchmark_results = {}

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(f"/Shared/scf_cohort/{catalog}_hyperparameter_tuning")

# Parent MLflow run — one per training cycle
with mlflow.start_run(run_name=f"hpo_training_{cutoff_period}") as parent_run:
    mlflow.log_param("cutoff_period",   cutoff_period)
    mlflow.log_param("param_grid_size", len(PARAM_GRID))
    mlflow.log_param("cv_folds",        len(cv_cutoffs))
    mlflow.log_param("cv_cutoffs",      str(cv_cutoffs))

    for product in agg_df['product'].unique():
        print(f"\nProduct: {product}")
        best_mape   = float("inf")
        best_result = None
        best_params = PARAM_GRID[0]
        safe_name   = product.replace(" ", "_").replace("(", "").replace(")", "")

        # ── Baseline benchmark ─────────────────────────────────────────
        b_mape = baseline_mape(agg_df, product)
        benchmark_results[product] = b_mape
        print(f"  Baseline MAPE (naive last-value): {b_mape:.3f}")
        mlflow.log_metric(f"{safe_name}__baseline_mape", b_mape)

        # ── Grid search with nested child runs ────────────────────────
        for params in PARAM_GRID:
            # Each trial = one child run in MLflow → visible in UI
            with mlflow.start_run(
                run_name=f"{safe_name}__mend{params['month_end']}_dm2{params['drop_month2']}",
                nested=True,
            ) as child_run:
                mlflow.log_param("product",       product)
                mlflow.log_param("month_end",     params["month_end"])
                mlflow.log_param("drop_month2",   params["drop_month2"])
                mlflow.log_param("cutoff_period", cutoff_period)

                # Rolling-origin CV score
                cv_mape = rolling_origin_cv(
                    pipeline, product, agg_df, cv_cutoffs,
                    params["month_end"], params["drop_month2"]
                )
                mlflow.log_metric("cv_mape", cv_mape)

                # Final fit on full training window
                try:
                    result = pipeline.run_for_product(
                        PRODUCT=product,
                        cutoff_period=cutoff_period,
                        drop_month2=params["drop_month2"],
                        month_start=2,
                        month_end=params["month_end"],
                        proj_start_period=cutoff_period,
                        proj_end_period=str(agg_df['reporting_period'].max()),
                        proj_cohort_cutoff='2024-12',
                        save_csv=False,
                    )
                    metrics = result.get("metrics") or {}
                    val_mape = metrics.get("mape_balance", float("inf"))
                    mlflow.log_metric("val_mape",  val_mape)
                    mlflow.log_metric("val_rmse",  metrics.get("rmse_balance", float("inf")))
                    print(f"  {params}  cv={cv_mape:.3f}  val={val_mape:.3f}")
                    if val_mape < best_mape:
                        best_mape   = val_mape
                        best_result = result
                        best_params = params
                    mlflow.set_tag("trial_status", "SUCCESS")
                except Exception as e:
                    mlflow.set_tag("trial_status", f"FAILED: {e}")
                    print(f"  {params} FAILED: {e}")

        if best_result is not None:
            all_results[product] = best_result
            best_params_per_product[product] = {**best_params, "best_mape": best_mape}
            # Log best config back to parent run
            mlflow.log_param(f"{safe_name}__best_month_end",    best_params["month_end"])
            mlflow.log_param(f"{safe_name}__best_drop_month2",  best_params["drop_month2"])
            mlflow.log_metric(f"{safe_name}__best_mape",        best_mape)
            # Benchmarking: did GLM beat the naive baseline?
            beats_baseline = best_mape < b_mape
            mlflow.log_metric(f"{safe_name}__beats_baseline",   int(beats_baseline))
            if not beats_baseline:
                print(f"  WARNING: GLM MAPE {best_mape:.3f} does not beat baseline {b_mape:.3f} for {product}")
            else:
                print(f"  Best: {best_params}  val_mape={best_mape:.3f}  vs baseline={b_mape:.3f} ✓")
        else:
            print(f"  WARNING: all param configs failed for {product}")

    # Portfolio-level benchmark summary
    products_beating_baseline = sum(
        1 for p, bp in best_params_per_product.items()
        if bp["best_mape"] < benchmark_results.get(p, float("inf"))
    )
    mlflow.log_metric("products_beating_baseline", products_beating_baseline)
    mlflow.log_metric("total_products_trained",    len(all_results))
    parent_run_id = parent_run.info.run_id

assert all_results, "No products trained successfully - aborting pipeline"
print(f"\nTrained {len(all_results)} / {agg_df['product'].nunique()} products")
print(f"Products beating naive baseline: {products_beating_baseline}/{len(all_results)}")

# COMMAND ----------
# Persist to a UC Volume so 04_model_evaluation.py / 05_model_registration.py
# can pick up the exact same fitted objects without retraining. UC Volumes
# are addressable as normal filesystem paths on UC-enabled clusters - no
# dbfs:/ prefix or dbutils.fs needed for plain Python file I/O.
VOLUME_PATH = f"/Volumes/{catalog}/{model_schema}/model_artifacts"
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{model_schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{model_schema}.model_artifacts")

artifact = {
    "results": all_results,
    "trained_at": datetime.datetime.utcnow().isoformat(),
    "cutoff_period": cutoff_period,
}
artifact_path = f"{VOLUME_PATH}/latest_training_run.pkl"
with open(artifact_path, "wb") as f:
    pickle.dump(artifact, f)

print(f"Saved training artifact to {artifact_path}")
