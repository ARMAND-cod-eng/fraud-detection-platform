# =============================================================
# test_model.py : Unit Tests for Model Scoring
# Production Fraud Detection Platform on AWS
# Author: Armand Junior Dongmo Notue
# =============================================================
# Run: pytest tests/test_model.py -v
# =============================================================

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
)))

from src.model import (
    find_best_threshold,
    evaluate_model,
    print_metrics
)


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def perfect_scores():
    """Perfect scores — fraud always scores 1.0."""
    y_true  = np.array([0, 0, 0, 1, 1, 1])
    y_scores = np.array([0.01, 0.02, 0.03,
                         0.99, 0.98, 0.97])
    return y_true, y_scores


@pytest.fixture
def mixed_scores():
    """Realistic mixed scores."""
    np.random.seed(42)
    y_true   = np.array([0]*90 + [1]*10)
    y_scores = np.concatenate([
        np.random.uniform(0.0, 0.3, 90),
        np.random.uniform(0.6, 1.0, 10)
    ])
    return y_true, y_scores


@pytest.fixture
def v3_simulation():
    """
    Simulated V3 model results matching
    the actual notebook output.
    """
    np.random.seed(42)
    n_legit = 113339
    n_fraud  = 2889 + 1244  # tp + fn

    legit_scores = np.random.beta(0.5, 5, n_legit)
    fraud_scores = np.concatenate([
        np.random.uniform(0.87, 1.0, 2889),  # TP
        np.random.uniform(0.0, 0.87, 1244)   # FN
    ])
    y_scores = np.concatenate(
        [legit_scores, fraud_scores]
    )
    y_true   = np.concatenate([
        np.zeros(n_legit),
        np.ones(n_legit - n_legit + n_fraud)
    ])
    return y_true[:len(y_scores)], y_scores


# ── Tests: find_best_threshold ────────────────────────────────

class TestFindBestThreshold:

    def test_returns_tuple(self, mixed_scores):
        """Must return a tuple of 2 values."""
        y_true, y_scores = mixed_scores
        result = find_best_threshold(
            y_true, y_scores
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_threshold_in_range(self, mixed_scores):
        """Threshold must be between 0.1 and 0.9."""
        y_true, y_scores = mixed_scores
        threshold, _ = find_best_threshold(
            y_true, y_scores
        )
        assert 0.1 <= threshold <= 0.9

    def test_f1_is_positive(self, mixed_scores):
        """Best F1 must be positive."""
        y_true, y_scores = mixed_scores
        _, best_f1 = find_best_threshold(
            y_true, y_scores
        )
        assert best_f1 > 0

    def test_perfect_separation(self, perfect_scores):
        """Perfect scores must give F1 = 1.0."""
        y_true, y_scores = perfect_scores
        threshold, f1 = find_best_threshold(
            y_true, y_scores
        )
        assert f1 == pytest.approx(1.0, abs=1e-5)

    def test_returns_floats(self, mixed_scores):
        """Both values must be floats."""
        y_true, y_scores = mixed_scores
        threshold, f1 = find_best_threshold(
            y_true, y_scores
        )
        assert isinstance(threshold, float)
        assert isinstance(f1, float)

    def test_custom_range(self, mixed_scores):
        """Custom range must be respected."""
        y_true, y_scores = mixed_scores
        threshold, _ = find_best_threshold(
            y_true, y_scores,
            start=0.5, end=0.9, step=0.01
        )
        assert 0.5 <= threshold <= 0.9


# ── Tests: evaluate_model ─────────────────────────────────────

class TestEvaluateModel:

    def test_returns_dict(self, mixed_scores):
        """Must return a dictionary."""
        y_true, y_scores = mixed_scores
        result = evaluate_model(y_true, y_scores)
        assert isinstance(result, dict)

    def test_all_keys_present(self, mixed_scores):
        """All expected keys must be present."""
        y_true, y_scores = mixed_scores
        result   = evaluate_model(y_true, y_scores)
        expected = [
            'auc_roc', 'auc_pr', 'f1',
            'precision', 'recall',
            'tp', 'fp', 'tn', 'fn', 'threshold'
        ]
        for key in expected:
            assert key in result, \
                f"Missing key: {key}"

    def test_metrics_in_range(self, mixed_scores):
        """All metrics must be between 0 and 1."""
        y_true, y_scores = mixed_scores
        result = evaluate_model(y_true, y_scores)
        for metric in [
            'auc_roc', 'auc_pr', 'f1',
            'precision', 'recall'
        ]:
            assert 0.0 <= result[metric] <= 1.0, \
                f"{metric} out of range: {result[metric]}"

    def test_confusion_matrix_sums(self, mixed_scores):
        """TP+FP+TN+FN must equal total samples."""
        y_true, y_scores = mixed_scores
        result = evaluate_model(y_true, y_scores)
        total  = (result['tp'] + result['fp'] +
                  result['tn'] + result['fn'])
        assert total == len(y_true)

    def test_perfect_model(self, perfect_scores):
        """Perfect model must have AUC-ROC = 1.0."""
        y_true, y_scores = perfect_scores
        result = evaluate_model(
            y_true, y_scores, threshold=0.5
        )
        assert result['auc_roc'] == \
            pytest.approx(1.0, abs=1e-5)

    def test_threshold_stored(self, mixed_scores):
        """Threshold must be stored in result."""
        y_true, y_scores = mixed_scores
        result = evaluate_model(
            y_true, y_scores, threshold=0.87
        )
        assert result['threshold'] == 0.87

    def test_integer_confusion_matrix(
        self, mixed_scores
    ):
        """Confusion matrix values must be integers."""
        y_true, y_scores = mixed_scores
        result = evaluate_model(y_true, y_scores)
        for key in ['tp', 'fp', 'tn', 'fn']:
            assert isinstance(result[key], int), \
                f"{key} must be int"

    def test_tp_plus_fn_equals_total_fraud(
        self, mixed_scores
    ):
        """TP + FN must equal total fraud cases."""
        y_true, y_scores = mixed_scores
        result     = evaluate_model(y_true, y_scores)
        total_fraud = int(y_true.sum())
        assert (result['tp'] + result['fn']
                ) == total_fraud

    def test_no_division_by_zero_all_legit(self):
        """Must handle all-legitimate predictions."""
        y_true   = np.zeros(100)
        y_true[0] = 1
        y_scores = np.zeros(100)
        # Should not raise ZeroDivisionError
        result = evaluate_model(y_true, y_scores)
        assert result['precision'] == 0.0
        assert result['recall']    == 0.0
        assert result['f1']        == 0.0


# ── Tests: print_metrics ──────────────────────────────────────

class TestPrintMetrics:

    def test_runs_without_error(
        self, mixed_scores, capsys
    ):
        """print_metrics must run without errors."""
        y_true, y_scores = mixed_scores
        metrics = evaluate_model(y_true, y_scores)
        print_metrics(metrics)
        captured = capsys.readouterr()
        assert 'AUC-ROC' in captured.out

    def test_prints_all_metrics(
        self, mixed_scores, capsys
    ):
        """All key metrics must appear in output."""
        y_true, y_scores = mixed_scores
        metrics = evaluate_model(y_true, y_scores)
        print_metrics(metrics)
        captured = capsys.readouterr()
        for label in [
            'AUC-ROC', 'AUC-PR', 'F1',
            'Precision', 'Recall'
        ]:
            assert label in captured.out, \
                f"Missing label: {label}"





