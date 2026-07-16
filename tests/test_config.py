"""
tests/test_config.py
Tests for src/common/config.py — verifies all UC table paths are correctly formed.
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.common.config import get_config


def test_config_table_paths_use_correct_catalog_and_schema():
    cfg = get_config("my_catalog", "my_schema")
    assert cfg.raw_base_data          == "my_catalog.my_schema.raw_base_data"
    assert cfg.silver_agg_cohort      == "my_catalog.my_schema.silver_agg_cohort"
    assert cfg.feature_store_table    == "my_catalog.my_schema.feature_store_cohort"
    assert cfg.inference_predictions  == "my_catalog.my_schema.inference_predictions"
    assert cfg.monitoring_metrics     == "my_catalog.my_schema.monitoring_metrics"
    assert cfg.drift_results          == "my_catalog.my_schema.drift_results"
    assert cfg.model_approvals        == "my_catalog.my_schema.model_approvals"
    assert cfg.model_cards            == "my_catalog.my_schema.model_cards"
    assert cfg.deployment_history     == "my_catalog.my_schema.deployment_history"
    assert cfg.rollback_events        == "my_catalog.my_schema.rollback_events"
    assert cfg.training_feature_snapshots == "my_catalog.my_schema.training_feature_snapshots"


def test_config_secret_scope_default():
    cfg = get_config("cat", "sch")
    assert cfg.secret_scope == "scf-cohort-notifications"


def test_config_secret_scope_override():
    cfg = get_config("cat", "sch", secret_scope="scf-cohort-prod")
    assert cfg.secret_scope == "scf-cohort-prod"


def test_config_thresholds_sensible():
    cfg = get_config("cat", "sch")
    assert 0 < cfg.PSI_WARN_THRESHOLD < cfg.PSI_ALERT_THRESHOLD
    assert cfg.RETRAIN_NULL_RATE_THRESHOLD > 0
    assert cfg.RETRAIN_MAPE_THRESHOLD > 0
    assert cfg.MAPE_PROMOTION_THRESHOLD > 0
