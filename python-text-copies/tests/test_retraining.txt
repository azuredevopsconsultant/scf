from src.common.retraining import evaluate_retraining_need


def test_retraining_trigger_on_drift():
    recommend, reason = evaluate_retraining_need(
        drift_detected=True,
        max_null_rate=0.01,
        portfolio_mape=5.0,
        null_rate_threshold=0.1,
        mape_threshold=20.0,
    )
    assert recommend is True
    assert "PSI alert" in reason


def test_retraining_trigger_on_null_rate():
    recommend, reason = evaluate_retraining_need(
        drift_detected=False,
        max_null_rate=0.2,
        portfolio_mape=5.0,
        null_rate_threshold=0.1,
        mape_threshold=20.0,
    )
    assert recommend is True
    assert "Null prediction rate" in reason


def test_retraining_trigger_none():
    recommend, reason = evaluate_retraining_need(
        drift_detected=False,
        max_null_rate=0.01,
        portfolio_mape=5.0,
        null_rate_threshold=0.1,
        mape_threshold=20.0,
    )
    assert recommend is False
    assert reason == "No retraining trigger crossed"
