"""
tests/test_drift_checks.py
Tests for src/common/drift_checks.py — run with: pytest tests/ (no cluster needed)
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from src.common.drift_checks import _psi_for_series, compute_drift_report, any_alerts


def test_psi_identical_distributions_is_zero():
    rng = np.random.default_rng(42)
    s = pd.Series(rng.normal(0, 1, 1000))
    psi = _psi_for_series(s, s)
    assert psi < 0.01  # identical → near-zero PSI


def test_psi_very_different_distributions_is_high():
    rng = np.random.default_rng(42)
    ref = pd.Series(rng.normal(0, 1, 1000))
    cur = pd.Series(rng.normal(5, 1, 1000))  # completely shifted
    psi = _psi_for_series(ref, cur)
    assert psi > 0.25  # large shift → ALERT-level PSI


def test_psi_empty_series_returns_nan():
    psi = _psi_for_series(pd.Series([], dtype=float), pd.Series([1.0, 2.0]))
    assert np.isnan(psi)


def test_compute_drift_report_returns_correct_status():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"int_rate": rng.normal(3.0, 0.1, 500)})
    cur = pd.DataFrame({"int_rate": rng.normal(3.0, 0.1, 500)})  # same dist
    report = compute_drift_report(ref, cur, ["int_rate"], warn_threshold=0.1, alert_threshold=0.25)
    assert len(report) == 1
    assert report["feature"].iloc[0] == "int_rate"
    assert report["status"].iloc[0] == "OK"


def test_compute_drift_report_raises_alert_on_shift():
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({"int_rate": rng.normal(1.0, 0.1, 500)})
    cur = pd.DataFrame({"int_rate": rng.normal(5.0, 0.1, 500)})  # big shift
    report = compute_drift_report(ref, cur, ["int_rate"], warn_threshold=0.1, alert_threshold=0.25)
    assert report["status"].iloc[0] == "ALERT"


def test_any_alerts_true_when_alert_present():
    report = pd.DataFrame({"feature": ["f1", "f2"], "psi": [0.3, 0.05], "status": ["ALERT", "OK"]})
    assert any_alerts(report) is True


def test_any_alerts_false_when_no_alert():
    report = pd.DataFrame({"feature": ["f1"], "psi": [0.05], "status": ["OK"]})
    assert any_alerts(report) is False


def test_psi_constant_series_returns_zero():
    s = pd.Series([1.0] * 100)
    psi = _psi_for_series(s, s)
    assert psi == 0.0  # not enough variance → returns 0
