"""
src/common/quality_checks.py

Lightweight, dependency-free data-quality checks (no Great Expectations /
Deequ requirement, so this runs on any cluster). Swap in Databricks Lakehouse
Monitoring or Great Expectations if your platform team standardises on one -
the check functions here just need to keep returning the same result shape.
"""
from pyspark.sql import DataFrame, functions as F


def check_row_count(df: DataFrame, expected_min: int) -> dict:
    count = df.count()
    return {
        "check": "row_count",
        "passed": count >= expected_min,
        "detail": f"{count} rows (expected >= {expected_min})",
    }


def check_null_rates(df: DataFrame, columns: list[str], max_null_rate: float) -> list[dict]:
    total = df.count()
    results = []
    for c in columns:
        nulls = df.filter(F.col(c).isNull()).count()
        rate = nulls / total if total else 1.0
        results.append(
            {
                "check": f"null_rate::{c}",
                "passed": rate <= max_null_rate,
                "detail": f"{rate:.2%} null (threshold {max_null_rate:.0%})",
            }
        )
    return results


def check_value_range(df: DataFrame, column: str, min_val: float, max_val: float) -> dict:
    out_of_range = df.filter((F.col(column) < min_val) | (F.col(column) > max_val)).count()
    return {
        "check": f"value_range::{column}",
        "passed": out_of_range == 0,
        "detail": f"{out_of_range} rows outside [{min_val}, {max_val}]",
    }


def check_schema_columns(df: DataFrame, required_columns: list[str]) -> dict:
    missing = [c for c in required_columns if c not in df.columns]
    return {
        "check": "schema_columns",
        "passed": len(missing) == 0,
        "detail": f"missing columns: {missing}" if missing else "all required columns present",
    }


def all_checks_passed(results: list[dict]) -> bool:
    return all(r["passed"] for r in results)
