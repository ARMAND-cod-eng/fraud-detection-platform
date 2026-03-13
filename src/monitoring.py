# =============================================================
# monitoring.py — Drift Detection + Alerting
# Production Fraud Detection Platform on AWS
# Author: Armand Junior Dongmo Notue
# =============================================================

import numpy as np
import pandas as pd
from scipy import stats


def ks_test(
    baseline: pd.Series,
    current: pd.Series
) -> tuple:
    """
    Kolmogorov-Smirnov test for distribution drift.

    Measures the maximum distance between two
    cumulative distributions.

    Parameters
    ----------
    baseline : pd.Series  Reference distribution
    current  : pd.Series  Current distribution

    Returns
    -------
    tuple : (ks_statistic, p_value)
            p_value < 0.05 = significant drift
    """
    stat, pval = stats.ks_2samp(
        baseline.fillna(0),
        current.fillna(0)
    )
    return float(stat), float(pval)


def psi(
    baseline: pd.Series,
    current: pd.Series,
    bins: int = 10
) -> float:
    """
    Population Stability Index (PSI).

    Industry standard thresholds:
        PSI < 0.10  -> No drift (stable)
        PSI 0.10-0.25 -> Moderate drift (monitor)
        PSI > 0.25  -> Severe drift (retrain!)

    Parameters
    ----------
    baseline : pd.Series  Reference distribution
    current  : pd.Series  Current distribution
    bins     : int        Number of bins (default 10)

    Returns
    -------
    float : PSI score
    """
    min_val     = min(baseline.min(), current.min())
    max_val     = max(baseline.max(), current.max())
    breakpoints = np.linspace(min_val, max_val, bins+1)

    base_counts = np.histogram(
        baseline, bins=breakpoints
    )[0]
    curr_counts = np.histogram(
        current, bins=breakpoints
    )[0]

    base_pct = (
        base_counts + 0.001
    ) / len(baseline)
    curr_pct = (
        curr_counts + 0.001
    ) / len(current)

    return float(np.sum(
        (curr_pct - base_pct) *
        np.log(curr_pct / base_pct)
    ))


def psi_label(val: float) -> str:
    """
    Convert PSI score to human-readable label.

    Parameters
    ----------
    val : float  PSI score

    Returns
    -------
    str : 'Stable', 'Moderate', or 'SEVERE'
    """
    if val < 0.10:
        return 'Stable'
    if val < 0.25:
        return 'Moderate'
    return 'SEVERE'


def compute_feature_drift(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    features: list
) -> dict:
    """
    Compute KS test and PSI for each feature.

    Parameters
    ----------
    baseline : pd.DataFrame  Reference month data
    current  : pd.DataFrame  Current month data
    features : list          Features to monitor

    Returns
    -------
    dict : Per-feature drift results
    """
    results = {}

    for feat in features:
        if feat not in baseline.columns:
            continue
        if feat not in current.columns:
            continue

        base_vals = baseline[feat].fillna(0)
        curr_vals = current[feat].fillna(0)

        ks_stat, ks_pval = ks_test(
            base_vals, curr_vals
        )
        psi_val = psi(base_vals, curr_vals)

        results[feat] = {
            'ks_stat' : ks_stat,
            'ks_pval' : ks_pval,
            'psi'     : psi_val,
            'drifted' : (ks_pval < 0.05
                         or psi_val > 0.10),
            'severity': psi_label(psi_val)
        }

    return results


def get_alert_level(
    auc: float,
    f1: float,
    max_psi: float,
    base_auc: float,
    base_f1: float
) -> tuple:
    """
    Determine monitoring alert level based on
    model performance and feature drift.

    Alert levels:
        GREEN  -> All metrics within thresholds
        ORANGE -> Monitor closely
        RED    -> Retrain model immediately!

    Parameters
    ----------
    auc      : float  Current AUC-ROC
    f1       : float  Current F1-Score
    max_psi  : float  Maximum PSI across features
    base_auc : float  Baseline AUC-ROC
    base_f1  : float  Baseline F1-Score

    Returns
    -------
    tuple : (alert_level, list_of_reasons)
    """
    reasons = []

    # RED conditions
    if auc < 0.90:
        reasons.append(
            f"AUC={auc:.4f} below 0.90"
        )
    if f1 < base_f1 * 0.80:
        reasons.append(
            f"F1={f1:.4f} dropped "
            f"{(1-f1/base_f1)*100:.0f}% from baseline"
        )
    if max_psi > 0.25:
        reasons.append(
            f"PSI={max_psi:.4f} SEVERE drift"
        )
    if reasons:
        return 'RED', reasons

    # ORANGE conditions
    if auc < 0.95:
        reasons.append(
            f"AUC={auc:.4f} below 0.95"
        )
    if f1 < base_f1 * 0.90:
        reasons.append(
            f"F1={f1:.4f} dropped "
            f"{(1-f1/base_f1)*100:.0f}% from baseline"
        )
    if max_psi > 0.10:
        reasons.append(
            f"PSI={max_psi:.4f} moderate drift"
        )
    if reasons:
        return 'ORANGE', reasons

    return 'GREEN', ['All metrics within thresholds']


def print_alert(
    month: str,
    level: str,
    reasons: list
) -> None:
    """
    Print a formatted monitoring alert.

    Parameters
    ----------
    month   : str   Month identifier
    level   : str   GREEN, ORANGE, or RED
    reasons : list  List of alert reasons
    """
    icons = {
        'GREEN' : 'OK',
        'ORANGE': 'WARNING',
        'RED'   : 'ALERT'
    }
    icon = icons.get(level, 'INFO')
    print(f"\n   [{icon}] {month} — {level}")
    for reason in reasons:
        print(f"      -> {reason}")


