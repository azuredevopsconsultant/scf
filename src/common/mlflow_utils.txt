"""
src/common/mlflow_utils.py

Champion/Challenger pattern implemented with Unity Catalog Model Registry
*aliases* (the "stages" API - Staging/Production - is deprecated by
Databricks in favour of aliases + tags, so this is the current best
practice, not a stylistic choice).

Lifecycle aliases used:
  @baseline   - first-ever registered version (reference point)
  @challenger - newest candidate model under evaluation
  @champion   - currently deployed model serving production inference

Promotion flow:
  train → @challenger → validate (MAPE check) → @champion  (approved)
                                               → archived   (rejected)
"""
import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature

mlflow.set_registry_uri("databricks-uc")


def log_model(pyfunc_model, artifact_path: str, input_example, run_name: str):
    """Log a custom pyfunc model (wraps the per-product GLMs) to the active run."""
    signature = infer_signature(input_example, pyfunc_model.predict(None, input_example))
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=pyfunc_model,
            signature=signature,
            input_example=input_example,
            pip_requirements=["statsmodels", "patsy", "pandas", "numpy"],
        )
        return run.info.run_id


def register_challenger(run_id: str, artifact_path: str, model_name: str) -> str:
    """Register a new model version in UC and point the 'challenger' alias at it."""
    client = MlflowClient()
    model_uri = f"runs:/{run_id}/{artifact_path}"
    mv = mlflow.register_model(model_uri=model_uri, name=model_name)
    client.set_registered_model_alias(model_name, "challenger", mv.version)
    client.set_model_version_tag(model_name, mv.version, "stage_history", "challenger")
    return mv.version


def get_alias_version(model_name: str, alias: str):
    client = MlflowClient()
    try:
        return client.get_model_version_by_alias(model_name, alias)
    except Exception:
        return None  # e.g. no champion exists yet (first ever run)


def get_alias_metrics(model_name: str, alias: str) -> dict:
    """Pull the logged eval metrics for whichever version currently holds `alias`."""
    client = MlflowClient()
    mv = get_alias_version(model_name, alias)
    if mv is None:
        return {}
    run = client.get_run(mv.run_id)
    return dict(run.data.metrics)


def promote_challenger_to_champion(model_name: str):
    """Move the 'champion' alias to whichever version currently holds 'challenger'."""
    client = MlflowClient()
    challenger_mv = client.get_model_version_by_alias(model_name, "challenger")
    client.set_registered_model_alias(model_name, "champion", challenger_mv.version)
    client.set_model_version_tag(
        model_name, challenger_mv.version, "stage_history", "champion"
    )
    return challenger_mv.version


def evaluate_promotion(
    challenger_metrics: dict,
    model_name: str,
    metric_key: str = "mape",
    lower_is_better: bool = True,
    improvement_threshold_pct: float = 15.0,
) -> dict:
    """
    Decide whether the newly trained challenger should be promoted to champion.

    Returns a dict describing the decision so model_registration.py can log it
    and the confirmation-mail task can report it. First-ever run always
    promotes (no champion to compare against).
    """
    champion_metrics = get_alias_metrics(model_name, "champion")
    challenger_value = challenger_metrics.get(metric_key)

    if not champion_metrics:
        return {
            "promoted": True,
            "reason": "No existing champion - challenger promoted by default.",
            "champion_metric": None,
            "challenger_metric": challenger_value,
        }

    champion_value = champion_metrics.get(metric_key)
    if champion_value in (None, 0):
        return {
            "promoted": True,
            "reason": "Champion has no comparable metric - promoting challenger.",
            "champion_metric": champion_value,
            "challenger_metric": challenger_value,
        }

    pct_change = (champion_value - challenger_value) / champion_value * 100
    if not lower_is_better:
        pct_change = -pct_change

    promoted = pct_change >= improvement_threshold_pct
    return {
        "promoted": promoted,
        "reason": (
            f"Challenger {metric_key}={challenger_value:.3f} vs "
            f"champion {metric_key}={champion_value:.3f} "
            f"({pct_change:.1f}% improvement, threshold {improvement_threshold_pct}%)."
        ),
        "champion_metric": champion_value,
        "challenger_metric": challenger_value,
    }


# ── Model lifecycle management ────────────────────────────────────────────────

def set_registered_model_description(model_name: str, description: str) -> None:
    """Set a human-readable description on the registered model (applies to all versions)."""
    client = MlflowClient()
    client.update_registered_model(name=model_name, description=description)


def set_version_description(model_name: str, version: str, description: str) -> None:
    """Set a description on a specific model version."""
    client = MlflowClient()
    client.update_model_version(name=model_name, version=str(version), description=description)


def set_rich_version_tags(
    model_name: str,
    version: str,
    trained_at: str,
    cutoff_period: str,
    n_products: int,
    portfolio_mape: float,
    catalog: str,
    schema: str,
) -> None:
    """
    Tag a model version with rich metadata so Unity Catalog Explorer shows
    training provenance at a glance. Tags are visible in the UC UI under
    Catalog > model_name > version.
    """
    client = MlflowClient()
    tags = {
        "trained_at":       trained_at,
        "cutoff_period":    cutoff_period,
        "n_products":       str(n_products),
        "portfolio_mape":   f"{portfolio_mape:.4f}",
        "catalog":          catalog,
        "schema":           schema,
        "model_type":       "GLM_cohort_projection",
        "framework":        "statsmodels",
        "stage_history":    "challenger",
    }
    for key, value in tags.items():
        client.set_model_version_tag(model_name, str(version), key, value)


def archive_rejected_challenger(model_name: str) -> None:
    """
    If a previous @challenger version was NOT promoted to @champion (rejected),
    tag it as 'archived' and remove the @challenger alias so only the new
    challenger holds that alias. Mirrors the Rejected → Archived flow in
    the Databricks MLOps lifecycle diagram.
    """
    client = MlflowClient()
    try:
        old_challenger = client.get_model_version_by_alias(model_name, "challenger")
        champion = get_alias_version(model_name, "champion")
        # Only archive if it wasn't promoted to champion
        if champion is None or old_challenger.version != champion.version:
            client.set_model_version_tag(
                model_name, old_challenger.version, "stage_history", "archived"
            )
            client.set_model_version_tag(
                model_name, old_challenger.version, "archived_reason", "superseded_by_new_challenger"
            )
            print(f"Archived previous challenger v{old_challenger.version}")
    except Exception:
        pass  # No previous challenger - first run


def set_baseline_alias(model_name: str, version: str) -> None:
    """
    Tag the very first registered version as @baseline. Baseline is the
    reference point for all future Champion/Challenger comparisons.
    Only applied once - if @baseline already exists, this is a no-op.
    """
    client = MlflowClient()
    existing = get_alias_version(model_name, "baseline")
    if existing is None:
        client.set_registered_model_alias(model_name, "baseline", str(version))
        client.set_model_version_tag(model_name, str(version), "stage_history", "baseline")
        print(f"Set v{version} as @baseline (first-ever registered version)")


def search_best_run(experiment_name: str, metric_key: str = "portfolio_mape_balance") -> str:
    """
    Search MLflow experiment runs and return the run_id of the run with the
    best (lowest) value of metric_key. Used to programmatically select the
    best training run to register, rather than always taking the latest run.
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment not found: {experiment_name}")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"metrics.{metric_key} > 0",
        order_by=[f"metrics.{metric_key} ASC"],
        max_results=1,
    )
    if not runs:
        raise ValueError(f"No runs found with metric '{metric_key}' in {experiment_name}")
    best_run = runs[0]
    print(
        f"Best run: {best_run.info.run_id}  "
        f"{metric_key}={best_run.data.metrics.get(metric_key, 'N/A'):.4f}"
    )
    return best_run.info.run_id

