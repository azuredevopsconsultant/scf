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
dbutils.widgets.text("catalog", "poc_mlops_dev")
dbutils.widgets.text("schema", "savings_cashflow")
dbutils.widgets.text("model_schema", "ml_models")
dbutils.widgets.text("training_start_period", "2022-12")
dbutils.widgets.text("cutoff_period", "2026-01")
dbutils.widgets.text("test_end_period", "2026-05")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_schema = dbutils.widgets.get("model_schema")
training_start_period = dbutils.widgets.get("training_start_period")
cutoff_period = dbutils.widgets.get("cutoff_period")
test_end_period = dbutils.widgets.get("test_end_period")

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
import pandas as pd
import math


def safe_log_metric(key, value):
    """MLflow rejects NaN/Inf metric values (aborts the task). Skip them so a
    product with no scoreable fold doesn't kill the whole run."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return
    if math.isfinite(v):
        mlflow.log_metric(key, v)


def rolling_origin_cv(
    pipeline,
    product,
    df,
    cutoffs,
    month_end,
    drop_month2,
    training_start_period,
):
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
                training_start_period=training_start_period,
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
training_start = Period(training_start_period, freq='M')
test_end = Period(test_end_period, freq='M')
assert training_start < main_cutoff <= test_end, (
    "Expected training_start_period < cutoff_period <= test_end_period"
)
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
    mlflow.log_param("training_start_period", training_start_period)
    mlflow.log_param("cutoff_period",   cutoff_period)
    mlflow.log_param("test_end_period", test_end_period)
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
        safe_log_metric(f"{safe_name}__baseline_mape", b_mape)

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
                    params["month_end"], params["drop_month2"],
                    training_start_period,
                )
                safe_log_metric("cv_mape", cv_mape)

                # Final fit on full training window
                try:
                    result = pipeline.run_for_product(
                        PRODUCT=product,
                        cutoff_period=cutoff_period,
                        drop_month2=params["drop_month2"],
                        month_start=2,
                        month_end=params["month_end"],
                        proj_start_period=cutoff_period,
                        proj_end_period=test_end_period,
                        proj_cohort_cutoff='2024-12',
                        save_csv=False,
                        training_start_period=training_start_period,
                    )
                    metrics = result.get("metrics") or {}
                    val_mape = metrics.get("mape_balance", float("inf"))
                    safe_log_metric("val_mape",  val_mape)
                    safe_log_metric("val_rmse",  metrics.get("rmse_balance", float("inf")))

                    # GLM-native goodness-of-fit for each of the three sub-models
                    # (deviance / AIC / pseudo-R2). These judge the ratio models
                    # themselves, complementing the reconstructed-balance MAPE.
                    for res_key, tag in (
                        ("rec_results", "rec"),
                        ("outflow_results", "outflow"),
                        ("wd_results", "wd"),
                    ):
                        gres = result.get(res_key)
                        if gres is None:
                            continue
                        try:
                            safe_log_metric(f"{tag}_deviance", float(gres.deviance))
                            safe_log_metric(f"{tag}_aic", float(gres.aic))
                            if getattr(gres, "null_deviance", None):
                                pseudo_r2 = 1.0 - float(gres.deviance) / float(gres.null_deviance)
                                safe_log_metric(f"{tag}_pseudo_r2", pseudo_r2)
                        except Exception:
                            pass

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
            safe_log_metric(f"{safe_name}__best_mape",        best_mape)
            # Benchmarking: did GLM beat the naive baseline?
            beats_baseline = best_mape < b_mape
            safe_log_metric(f"{safe_name}__beats_baseline",   int(beats_baseline))
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

    # ── Model-risk sign-off: per-product GLM coefficient summary ──────────
    # One row per (product × sub-model × term) with estimate, robust SE,
    # p-value, CI and per-model goodness-of-fit. Logged as an MLflow artifact
    # on the parent run and persisted to the model_coefficients audit table so
    # second-line reviewers can inspect exactly what each GLM learned without
    # unpickling the fitted objects.
    coef_rows = []
    for product, result in all_results.items():
        for res_key, sub in (
            ("rec_results", "receipts"),
            ("outflow_results", "outflow"),
            ("wd_results", "withdrawal_split"),
        ):
            gres = result.get(res_key)
            if gres is None:
                continue
            try:
                conf = gres.conf_int()
                deviance = float(getattr(gres, "deviance", float("nan")))
                null_dev = getattr(gres, "null_deviance", None)
                pseudo_r2 = (1.0 - deviance / float(null_dev)) if null_dev else float("nan")
                converged = bool(
                    getattr(gres, "converged", None)
                    if getattr(gres, "converged", None) is not None
                    else getattr(gres, "mle_retvals", {}).get("converged", True)
                )
                for term in gres.params.index:
                    pval = float(gres.pvalues[term])
                    coef_rows.append({
                        "product": product,
                        "sub_model": sub,
                        "family": type(gres.model.family).__name__,
                        "term": str(term),
                        "coefficient": float(gres.params[term]),
                        "std_err": float(gres.bse[term]),
                        "p_value": pval,
                        "ci_lower": float(conf.loc[term].iloc[0]),
                        "ci_upper": float(conf.loc[term].iloc[1]),
                        "significant_5pct": bool(pval < 0.05),
                        "deviance": deviance,
                        "aic": float(getattr(gres, "aic", float("nan"))),
                        "pseudo_r2": pseudo_r2,
                        "n_obs": int(getattr(gres, "nobs", 0)),
                        "converged": converged,
                    })
            except Exception as e:
                print(f"  coefficient extraction failed for {product}/{sub}: {e}")

    coef_df = pd.DataFrame(coef_rows)
    coef_df["cutoff_period"] = cutoff_period
    coef_df["trained_at"] = datetime.datetime.utcnow().isoformat()
    if not coef_df.empty:
        coef_csv = f"/tmp/glm_coefficients_{cutoff_period}.csv"
        coef_df.to_csv(coef_csv, index=False)
        mlflow.log_artifact(coef_csv, artifact_path="model_risk")
        print(f"Logged coefficient summary artifact: {len(coef_df)} rows across {coef_df['product'].nunique()} products")

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
    "training_start_period": training_start_period,
    "cutoff_period": cutoff_period,
    "test_end_period": test_end_period,
}
artifact_path = f"{VOLUME_PATH}/latest_training_run.pkl"
with open(artifact_path, "wb") as f:
    pickle.dump(artifact, f)

print(f"Saved training artifact to {artifact_path}")

# COMMAND ----------
# Persist the coefficient summary to the model_coefficients audit table so
# second-line reviewers can query per-product GLM estimates for sign-off
# alongside the MLflow artifact logged on the parent run.
if not coef_df.empty:
    from pyspark.sql.types import (
        StructType, StructField, StringType, DoubleType, BooleanType, LongType,
    )

    coef_schema = StructType([
        StructField("product",          StringType(),  True),
        StructField("sub_model",        StringType(),  True),
        StructField("family",           StringType(),  True),
        StructField("term",             StringType(),  True),
        StructField("coefficient",      DoubleType(),  True),
        StructField("std_err",          DoubleType(),  True),
        StructField("p_value",          DoubleType(),  True),
        StructField("ci_lower",         DoubleType(),  True),
        StructField("ci_upper",         DoubleType(),  True),
        StructField("significant_5pct", BooleanType(), True),
        StructField("deviance",         DoubleType(),  True),
        StructField("aic",              DoubleType(),  True),
        StructField("pseudo_r2",        DoubleType(),  True),
        StructField("n_obs",            LongType(),    True),
        StructField("converged",        BooleanType(), True),
        StructField("cutoff_period",    StringType(),  True),
        StructField("trained_at",       StringType(),  True),
    ])

    # Non-finite doubles (NaN/Inf) -> NULL so the typed Delta append never
    # trips MLflow-style value rejection or Arrow schema-merge errors.
    coef_out = coef_df.reindex(columns=[f.name for f in coef_schema]).copy()
    for c in ["coefficient", "std_err", "p_value", "ci_lower", "ci_upper",
              "deviance", "aic", "pseudo_r2"]:
        s = pd.to_numeric(coef_out[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        coef_out[c] = s.astype(object).where(s.notna(), None)
    coef_out["n_obs"] = pd.to_numeric(coef_out["n_obs"], errors="coerce").fillna(0).astype("int64")
    coef_out["significant_5pct"] = coef_out["significant_5pct"].astype(bool)
    coef_out["converged"] = coef_out["converged"].astype(bool)

    spark.createDataFrame(coef_out, schema=coef_schema).write.mode("append").option(
        "mergeSchema", "true"
    ).saveAsTable(cfg.model_coefficients)
    print(f"Wrote {len(coef_out)} coefficient rows to {cfg.model_coefficients}")
