# =============================================================
# test_monitoring.py : Unit Tests for Drift Detection
# Production Fraud Detection Platform on AWS
# Author: Armand Junior Dongmo Notue
# =============================================================
# Run: pytest tests/test_monitoring.py -v
# =============================================================

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
)))

from src.monitoring import (
    ks_test,
    psi,
    psi_label,
    compute_feature_drift,
    get_alert_level
)


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def identical_distributions():
    """Two identical distributions — no drift."""
    np.random.seed(42)
    data = pd.Series(np.random.normal(0, 1, 1000))
    return data, data.copy()


@pytest.fixture
def drifted_distributions():
    """Two clearly different distributions."""
    np.random.seed(42)
    baseline = pd.Series(np.random.normal(0, 1, 1000))
    current  = pd.Series(np.random.normal(5, 1, 1000))
    return baseline, current


@pytest.fixture
def baseline_df():
    """Baseline month dataframe."""
    np.random.seed(42)
    n = 500
    return pd.DataFrame({
        'card1_fraud_rate': np.random.beta(2, 50, n),
        'TransactionAmt' : np.random.exponential(
            100, n
        ),
        'uid1_fraud_rate' : np.random.beta(2, 50, n),
    })


@pytest.fixture
def drifted_df(baseline_df):
    """Drifted month — card1_fraud_rate shifted."""
    np.random.seed(99)
    n      = 500
    drifted = baseline_df.copy()
    # Inject severe drift into card1_fraud_rate
    drifted['card1_fraud_rate'] = \
        np.random.beta(10, 10, n)
    return drifted


# ── Tests: ks_test ────────────────────────────────────────────

class TestKsTest:

    def test_returns_tuple(
        self, identical_distributions
    ):
        """Must return tuple of 2 floats."""
        base, curr = identical_distributions
        result     = ks_test(base, curr)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_identical_high_pvalue(
        self, identical_distributions
    ):
        """Identical distributions must have p > 0.05."""
        base, curr = identical_distributions
        _, p_value = ks_test(base, curr)
        assert p_value > 0.05

    def test_drifted_low_pvalue(
        self, drifted_distributions
    ):
        """Clearly drifted distributions: p < 0.05."""
        base, curr = drifted_distributions
        _, p_value = ks_test(base, curr)
        assert p_value < 0.05

    def test_statistic_range(
        self, identical_distributions
    ):
        """KS statistic must be between 0 and 1."""
        base, curr = identical_distributions
        stat, _    = ks_test(base, curr)
        assert 0.0 <= stat <= 1.0

    def test_handles_nan(self):
        """Must handle NaN values gracefully."""
        base = pd.Series([1.0, 2.0, np.nan, 4.0])
        curr = pd.Series([1.0, 2.0, 3.0, np.nan])
        stat, p_value = ks_test(base, curr)
        assert not np.isnan(stat)
        assert not np.isnan(p_value)


# ── Tests: psi ────────────────────────────────────────────────

class TestPsi:

    def test_returns_float(
        self, identical_distributions
    ):
        """Must return a float."""
        base, curr = identical_distributions
        result     = psi(base, curr)
        assert isinstance(result, float)

    def test_identical_near_zero(
        self, identical_distributions
    ):
        """Identical distributions must have PSI ≈ 0."""
        base, curr = identical_distributions
        result     = psi(base, curr)
        assert result < 0.05

    def test_drifted_above_threshold(
        self, drifted_distributions
    ):
        """Clearly drifted distributions: PSI > 0.25."""
        base, curr = drifted_distributions
        result     = psi(base, curr)
        assert result > 0.25

    def test_positive_value(
        self, identical_distributions
    ):
        """PSI must always be non-negative."""
        base, curr = identical_distributions
        result     = psi(base, curr)
        assert result >= 0.0

    def test_custom_bins(
        self, identical_distributions
    ):
        """Custom bins parameter must work."""
        base, curr = identical_distributions
        result_5   = psi(base, curr, bins=5)
        result_20  = psi(base, curr, bins=20)
        assert isinstance(result_5, float)
        assert isinstance(result_20, float)


# ── Tests: psi_label ─────────────────────────────────────────

class TestPsiLabel:

    def test_stable_label(self):
        """PSI < 0.10 must return Stable."""
        assert psi_label(0.05) == 'Stable'
        assert psi_label(0.09) == 'Stable'

    def test_moderate_label(self):
        """PSI 0.10-0.25 must return Moderate."""
        assert psi_label(0.10) == 'Moderate'
        assert psi_label(0.20) == 'Moderate'
        assert psi_label(0.24) == 'Moderate'

    def test_severe_label(self):
        """PSI > 0.25 must return SEVERE."""
        assert psi_label(0.25) == 'SEVERE'
        assert psi_label(0.33) == 'SEVERE'
        assert psi_label(1.00) == 'SEVERE'

    def test_returns_string(self):
        """Must return a string."""
        assert isinstance(psi_label(0.05), str)
        assert isinstance(psi_label(0.15), str)
        assert isinstance(psi_label(0.30), str)


# ── Tests: compute_feature_drift ─────────────────────────────

class TestComputeFeatureDrift:

    def test_returns_dict(
        self, baseline_df, drifted_df
    ):
        """Must return a dictionary."""
        features = ['card1_fraud_rate',
                    'TransactionAmt']
        result   = compute_feature_drift(
            baseline_df, drifted_df, features
        )
        assert isinstance(result, dict)

    def test_all_features_present(
        self, baseline_df, drifted_df
    ):
        """All features must be in result."""
        features = ['card1_fraud_rate',
                    'TransactionAmt']
        result   = compute_feature_drift(
            baseline_df, drifted_df, features
        )
        for feat in features:
            assert feat in result

    def test_result_keys(
        self, baseline_df, drifted_df
    ):
        """Each feature must have required keys."""
        features = ['card1_fraud_rate']
        result   = compute_feature_drift(
            baseline_df, drifted_df, features
        )
        expected_keys = [
            'ks_stat', 'ks_pval', 'psi',
            'drifted', 'severity'
        ]
        for key in expected_keys:
            assert key in result['card1_fraud_rate']

    def test_drifted_feature_detected(
        self, baseline_df, drifted_df
    ):
        """Severely drifted feature must be flagged."""
        features = ['card1_fraud_rate']
        result   = compute_feature_drift(
            baseline_df, drifted_df, features
        )
        assert result['card1_fraud_rate']['drifted']

    def test_missing_feature_skipped(
        self, baseline_df, drifted_df
    ):
        """Features not in data must be skipped."""
        features = ['nonexistent_feature']
        result   = compute_feature_drift(
            baseline_df, drifted_df, features
        )
        assert 'nonexistent_feature' not in result


# ── Tests: get_alert_level ────────────────────────────────────

class TestGetAlertLevel:

    def test_green_alert(self):
        """Good metrics must return GREEN."""
        level, reasons = get_alert_level(
            auc=0.9976, f1=0.8363,
            max_psi=0.05,
            base_auc=0.9976, base_f1=0.8363
        )
        assert level == 'GREEN'

    def test_orange_alert_psi(self):
        """Moderate PSI must return ORANGE."""
        level, reasons = get_alert_level(
            auc=0.9960, f1=0.8358,
            max_psi=0.15,
            base_auc=0.9976, base_f1=0.8363
        )
        assert level == 'ORANGE'

    def test_red_alert_severe_psi(self):
        """Severe PSI must return RED."""
        level, reasons = get_alert_level(
            auc=0.9965, f1=0.8330,
            max_psi=0.33,
            base_auc=0.9976, base_f1=0.8363
        )
        assert level == 'RED'

    def test_red_alert_f1_drop(self):
        """40% F1 drop must return RED."""
        level, reasons = get_alert_level(
            auc=0.9058, f1=0.5058,
            max_psi=0.16,
            base_auc=0.9976, base_f1=0.8363
        )
        assert level == 'RED'

    def test_red_alert_auc_drop(self):
        """AUC below 0.90 must return RED."""
        level, reasons = get_alert_level(
            auc=0.88, f1=0.80,
            max_psi=0.05,
            base_auc=0.9976, base_f1=0.8363
        )
        assert level == 'RED'

    def test_returns_tuple(self):
        """Must return tuple of (str, list)."""
        result = get_alert_level(
            0.99, 0.83, 0.05, 0.99, 0.83
        )
        assert isinstance(result, tuple)
        assert isinstance(result[0], str)
        assert isinstance(result[1], list)

    def test_reasons_not_empty(self):
        """Reasons list must never be empty."""
        _, reasons = get_alert_level(
            0.99, 0.83, 0.05, 0.99, 0.83
        )
        assert len(reasons) > 0

    def test_matches_notebook_month5(self):
        """Month 5 must trigger RED (PSI=0.33)."""
        level, _ = get_alert_level(
            auc=0.9965, f1=0.8330,
            max_psi=0.3348,
            base_auc=0.9976, base_f1=0.8363
        )
        assert level == 'RED'

    def test_matches_notebook_month1(self):
        """Month 1 baseline must be GREEN."""
        level, _ = get_alert_level(
            auc=0.9976, f1=0.8363,
            max_psi=0.0,
            base_auc=0.9976, base_f1=0.8363
        )
        assert level == 'GREEN'



