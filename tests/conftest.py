# integration_test.py is a Databricks notebook that requires the Databricks
# runtime (dbutils, spark, etc.). Exclude it from local/CI pytest collection.
collect_ignore = ["integration_test.py"]
