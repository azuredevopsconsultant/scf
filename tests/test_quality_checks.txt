"""
tests/test_quality_checks.py
Tests for src/common/quality_checks.py — run with: pytest tests/ (no cluster needed)
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

from src.common.quality_checks import (
    check_row_count, check_null_rates, check_schema_columns,
    check_value_range, all_checks_passed,
)

@pytest.fixture(scope="module")
def spark():
    return SparkSession.builder.master("local[1]").appName("test_quality").getOrCreate()

@pytest.fixture
def good_df(spark):
    data = [("cohort_A", "EA ISA", 1.0, 0.8), ("cohort_B", "EA ISA", 2.0, 0.9)]
    schema = StructType([
        StructField("cohort", StringType()),
        StructField("product", StringType()),
        StructField("balance", DoubleType()),
        StructField("outflow_prop", DoubleType()),
    ])
    return spark.createDataFrame(data, schema)


def test_row_count_passes(good_df):
    result = check_row_count(good_df, expected_min=1)
    assert result["passed"] is True


def test_row_count_fails(good_df):
    result = check_row_count(good_df, expected_min=100)
    assert result["passed"] is False


def test_null_rates_pass_when_no_nulls(good_df):
    results = check_null_rates(good_df, ["balance", "outflow_prop"], max_null_rate=0.05)
    assert all(r["passed"] for r in results)


def test_null_rates_fail_with_nulls(spark):
    data = [(None, 1.0), ("cohort_B", 2.0)]
    schema = StructType([StructField("cohort", StringType()), StructField("balance", DoubleType())])
    df = spark.createDataFrame(data, schema)
    results = check_null_rates(df, ["cohort"], max_null_rate=0.01)
    assert not results[0]["passed"]


def test_schema_columns_pass(good_df):
    result = check_schema_columns(good_df, ["cohort", "product", "balance"])
    assert result["passed"] is True


def test_schema_columns_fail_on_missing(good_df):
    result = check_schema_columns(good_df, ["cohort", "missing_column"])
    assert result["passed"] is False
    assert "missing_column" in result["detail"]


def test_value_range_pass(good_df):
    result = check_value_range(good_df, "outflow_prop", 0.0, 1.0)
    assert result["passed"] is True


def test_value_range_fail_on_out_of_range(spark):
    data = [(1.5,), (0.5,)]
    schema = StructType([StructField("outflow_prop", DoubleType())])
    df = spark.createDataFrame(data, schema)
    result = check_value_range(df, "outflow_prop", 0.0, 1.0)
    assert result["passed"] is False


def test_all_checks_passed_with_all_true():
    checks = [{"passed": True}, {"passed": True}]
    assert all_checks_passed(checks) is True


def test_all_checks_passed_with_one_false():
    checks = [{"passed": True}, {"passed": False}]
    assert all_checks_passed(checks) is False
