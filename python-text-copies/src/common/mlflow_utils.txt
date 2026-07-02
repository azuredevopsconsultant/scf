"""
src/common/mlflow_utils.py

Champion/Challenger pattern implemented with Unity Catalog Model Registry
*aliases* (the "stages" API - Staging/Production - is deprecated by
Databricks in favour of aliases + tags, so this is the current best
practice, not a stylistic choice).

Convention used here:
  - Every successful training run is logged and registered as a new model
    version.
  - It is always tagged "challenger" (alias moved to the newest candidate).
  - It is ONLY promoted to "champion" if model_evaluation shows it beats the
    current champion by MAPE_PROMOTION_THRESHOLD (relative %). This keeps a
    human-auditable trail (git-style: every version exists, promotion is an
    explicit, logged event) and avoids a bad retrain silently taking over
    production inference.
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
