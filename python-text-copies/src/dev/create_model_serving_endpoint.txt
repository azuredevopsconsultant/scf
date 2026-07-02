"""Create or update a Databricks Model Serving endpoint for the champion model."""
from __future__ import annotations

import argparse

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
import mlflow


def get_champion_version(model_name: str) -> str:
    client = mlflow.MlflowClient(registry_uri="databricks-uc")
    version = client.get_model_version_by_alias(model_name, "champion").version
    return str(version)


def upsert_endpoint(endpoint_name: str, model_name: str) -> None:
    w = WorkspaceClient()
    champion_version = get_champion_version(model_name)
    config = EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name=model_name,
                entity_version=champion_version,
                workload_size="Small",
                scale_to_zero_enabled=True,
            )
        ]
    )

    try:
        w.serving_endpoints.get(endpoint_name)
        w.serving_endpoints.update_config_and_wait(name=endpoint_name, served_entities=config.served_entities)
        print(f"Updated endpoint {endpoint_name} -> {model_name} v{champion_version}")
    except Exception:
        w.serving_endpoints.create_and_wait(name=endpoint_name, config=config)
        print(f"Created endpoint {endpoint_name} -> {model_name} v{champion_version}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--endpoint-name", default="scf-cohort-serving-endpoint")
    args = parser.parse_args()
    upsert_endpoint(args.endpoint_name, args.model_name)


if __name__ == "__main__":
    main()
