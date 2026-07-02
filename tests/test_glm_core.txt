"""
tests/test_glm_core.py

Run with: pytest tests/ (no Databricks cluster needed - pure pandas/statsmodels)
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from src.common.glm_core import DataPrep, Evaluator, RATE_BINS, RATE_LABELS


def _make_fake_agg_df():
    periods = pd.period_range("2024-01", periods=6, freq="M")
    rows = []
    for cohort in periods[:2]:
        for i, rp in enumerate(periods):
            if rp < cohort:
                continue
            rows.append({
                "product": "EA ISA (Online)",
                "cohort": cohort,
                "reporting_period": rp,
                "months_since_start": (rp - cohort).n + 1,
                "balance": 1000 + i * 50,
                "balance_lag_1": 1000 + max(i - 1, 0) * 50,
                "receipts": 100,
                "withdrawals": 50,
                "transfers": 10,
                "rec_prop": 0.1,
                "outflow_prop": 0.05,
                "withdrawal_prop_of_outflow": 0.8,
                "delta_to_best_buy": -0.1,
                "dbb_range": "[-0.1, -0.05)",
            })
    return pd.DataFrame(rows)


def test_data_prep_split_respects_cutoff():
    df = _make_fake_agg_df()
    prep = DataPrep(df, product="EA ISA (Online)", cutoff_period="2024-04", drop_month2=False)
    train, test = prep.split_train_test()
    assert (train["reporting_period"] < pd.Period("2024-04", freq="M")).all()
    assert (test["reporting_period"] >= pd.Period("2024-04", freq="M")).all()


def test_data_prep_drops_month_2():
    df = _make_fake_agg_df()
    prep = DataPrep(df, product="EA ISA (Online)", cutoff_period="2024-06", drop_month2=True)
    train, _ = prep.split_train_test()
    assert 2 not in train["months_since_start"].values


def test_evaluator_mape_zero_error():
    evaluator = Evaluator()
    df = pd.DataFrame({"actual": [100, 200], "pred": [100, 200]})
    assert evaluator.mape(df, "actual", "pred") == 0


def test_evaluator_rmse_known_value():
    evaluator = Evaluator()
    df = pd.DataFrame({"actual": [0, 0], "pred": [3, 4]})
    assert evaluator.rmse(df, "actual", "pred") == pytest.approx(np.sqrt((9 + 16) / 2))


def test_rate_bins_labels_same_length():
    assert len(RATE_BINS) - 1 == len(RATE_LABELS)
