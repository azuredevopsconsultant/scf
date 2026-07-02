"""Helpers for writing auditable dataset version metadata to Unity Catalog."""
from __future__ import annotations

from typing import Any


def _escape_table_identifier(table_name: str) -> str:
    return table_name.replace("`", "``")


def _as_dict(row: Any) -> dict[str, Any]:
    if hasattr(row, "asDict"):
        return row.asDict(recursive=True)
    if isinstance(row, dict):
        return row
    return {}


def record_dataset_version(
    spark,
    source_table: str,
    tracking_table: str,
    dataset_name: str,
    environment: str,
) -> None:
    escaped_source = _escape_table_identifier(source_table)
    history = spark.sql(f"DESCRIBE HISTORY `{escaped_source}` LIMIT 1").collect()
    if not history:
        return

    h = _as_dict(history[0])
    params = h.get("operationParameters") or {}

    payload = [
        {
            "dataset_name": dataset_name,
            "source_table": source_table,
            "delta_version": str(h.get("version", "")),
            "operation": str(h.get("operation", "")),
            "operation_parameters": str(params),
            "recorded_at": str(h.get("timestamp", "")),
            "environment": environment,
        }
    ]

    spark.createDataFrame(payload).write.mode("append").option(
        "mergeSchema", "true"
    ).saveAsTable(tracking_table)
