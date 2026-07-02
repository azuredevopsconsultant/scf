"""Retraining recommendation logic used by inference monitoring workflows."""
from __future__ import annotations


def evaluate_retraining_need(
    drift_detected: bool,
    max_null_rate: float,
    portfolio_mape: float | None,
    null_rate_threshold: float,
    mape_threshold: float,
) -> tuple[bool, str]:
    reasons: list[str] = []

    if drift_detected:
        reasons.append("PSI alert detected")
    if max_null_rate > null_rate_threshold:
        reasons.append(
            f"Null prediction rate {max_null_rate:.2%} > threshold {null_rate_threshold:.2%}"
        )
    if portfolio_mape is not None and portfolio_mape > mape_threshold:
        reasons.append(
            f"Portfolio MAPE {portfolio_mape:.2f} > threshold {mape_threshold:.2f}"
        )

    if reasons:
        return True, "; ".join(reasons)
    return False, "No retraining trigger crossed"
