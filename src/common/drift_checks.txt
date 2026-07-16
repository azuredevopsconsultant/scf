"""
src/common/drift_checks.py

Population Stability Index (PSI) drift detection between a reference
(training-time) feature distribution and the current inference batch.
PSI is chosen over a raw KS-test because it's the standard banking /
credit-risk metric your model-risk-management team will already recognise,
and it degrades gracefully with binned continuous features like
`delta_to_best_buy`, `rec_prop`, etc.

Interpretation (industry-standard bands):
  PSI < 0.1              -> no significant shift
  0.1 <= PSI < 0.25       -> moderate shift, monitor
  PSI >= 0.25             -> significant shift, alert
"""
import numpy as np
import pandas as pd


def _psi_for_series(reference: pd.Series, current: pd.Series, buckets: int = 10) -> float:
    reference = reference.dropna()
    current = current.dropna()
    if reference.empty or current.empty:
        return np.nan

    breakpoints = np.unique(
        np.quantile(reference, np.linspace(0, 1, buckets + 1))
    )
    if len(breakpoints) < 3:
        return 0.0  # not enough variance to bucket meaningfully

    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current, bins=breakpoints)

    ref_pct = np.where(ref_counts == 0, 1e-4, ref_counts / ref_counts.sum())
    cur_pct = np.where(cur_counts == 0, 1e-4, cur_counts / cur_counts.sum())

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def compute_drift_report(
    reference_pdf: pd.DataFrame,
    current_pdf: pd.DataFrame,
    columns: list[str],
    warn_threshold: float = 0.1,
    alert_threshold: float = 0.25,
) -> pd.DataFrame:
    rows = []
    for col in columns:
        psi = _psi_for_series(reference_pdf[col], current_pdf[col])
        status = "OK"
        if psi >= alert_threshold:
            status = "ALERT"
        elif psi >= warn_threshold:
            status = "WARN"
        rows.append({"feature": col, "psi": psi, "status": status})
    return pd.DataFrame(rows)


def any_alerts(drift_report: pd.DataFrame) -> bool:
    return bool((drift_report["status"] == "ALERT").any())


def extrapolation_report(
    reference: pd.Series,
    current: pd.Series,
    feature: str,
    alert_rate: float = 0.05,
) -> dict:
    """
    GLM extrapolation guard. A GLM extrapolates linearly, so predictions made
    where a feature falls outside its training support are unreliable. Reports
    the fraction of the current batch outside the reference [min, max] range
    and flags ALERT when it exceeds ``alert_rate``.
    """
    reference = pd.to_numeric(reference, errors="coerce").dropna()
    current = pd.to_numeric(current, errors="coerce").dropna()
    if reference.empty or current.empty:
        return {"feature": feature, "ref_min": np.nan, "ref_max": np.nan,
                "out_of_range_rate": np.nan, "status": "OK"}

    ref_min, ref_max = float(reference.min()), float(reference.max())
    out_of_range = ((current < ref_min) | (current > ref_max)).mean()
    status = "ALERT" if out_of_range > alert_rate else "OK"
    return {
        "feature": feature,
        "ref_min": ref_min,
        "ref_max": ref_max,
        "out_of_range_rate": float(out_of_range),
        "status": status,
    }
