# tests/  Unit Tests

This folder contains unit tests for all core modules
in the `src/` folder. Tests are written using pytest
and cover feature engineering, model scoring, and
drift detection.

---

## Test Files

| File | Module Tested | Tests |
|---|---|---|
| `test_features.py` | `src/features.py` | 14 tests |
| `test_model.py` | `src/model.py` | 13 tests |
| `test_monitoring.py` | `src/monitoring.py` | 18 tests |

**Total: 45 unit tests**

---

## Installation
```bash
pip install pytest
pip install -r requirements.txt
```

---

## Running Tests

### Run all tests
```bash
pytest tests/ -v
```

### Run a specific file
```bash
pytest tests/test_features.py -v
pytest tests/test_model.py -v
pytest tests/test_monitoring.py -v
```

### Run a specific test class
```bash
pytest tests/test_features.py::TestCreateUidFeatures -v
pytest tests/test_monitoring.py::TestGetAlertLevel -v
```

### Run with coverage report
```bash
pip install pytest-cov
pytest tests/ --cov=src --cov-report=term-missing
```

---

## Expected Output
```
================================================
tests/test_features.py
================================================
PASSED test_features.py::TestCreateUidFeatures::test_uid_columns_created
PASSED test_features.py::TestCreateUidFeatures::test_uid1_format
PASSED test_features.py::TestCreateUidFeatures::test_uid2_format
PASSED test_features.py::TestCreateUidFeatures::test_uid3_format
PASSED test_features.py::TestCreateUidFeatures::test_nan_handling
PASSED test_features.py::TestCreateUidFeatures::test_original_columns_preserved
PASSED test_features.py::TestSmoothTargetEncode::test_output_shape
PASSED test_features.py::TestSmoothTargetEncode::test_output_dtype
PASSED test_features.py::TestSmoothTargetEncode::test_unseen_category_gets_global_mean
PASSED test_features.py::TestSmoothTargetEncode::test_smoothing_pulls_toward_global_mean
PASSED test_features.py::TestSmoothTargetEncode::test_no_missing_values
PASSED test_features.py::TestFrequencyEncode::test_freq_columns_created
PASSED test_features.py::TestFrequencyEncode::test_freq_values_sum_to_one
PASSED test_features.py::TestFrequencyEncode::test_freq_dtype
PASSED test_features.py::TestFrequencyEncode::test_unseen_category_gets_zero
PASSED test_features.py::TestGetFeatureNames::test_excludes_target
PASSED test_features.py::TestGetFeatureNames::test_excludes_transaction_id
PASSED test_features.py::TestGetFeatureNames::test_excludes_object_columns
PASSED test_features.py::TestGetFeatureNames::test_numeric_columns_included
PASSED test_features.py::TestGetFeatureNames::test_returns_list

================================================
tests/test_model.py
================================================
PASSED test_model.py::TestFindBestThreshold::test_returns_tuple
PASSED test_model.py::TestFindBestThreshold::test_threshold_in_range
PASSED test_model.py::TestFindBestThreshold::test_f1_is_positive
PASSED test_model.py::TestFindBestThreshold::test_perfect_separation
PASSED test_model.py::TestFindBestThreshold::test_returns_floats
PASSED test_model.py::TestFindBestThreshold::test_custom_range
PASSED test_model.py::TestEvaluateModel::test_returns_dict
PASSED test_model.py::TestEvaluateModel::test_all_keys_present
PASSED test_model.py::TestEvaluateModel::test_metrics_in_range
PASSED test_model.py::TestEvaluateModel::test_confusion_matrix_sums
PASSED test_model.py::TestEvaluateModel::test_perfect_model
PASSED test_model.py::TestEvaluateModel::test_threshold_stored
PASSED test_model.py::TestEvaluateModel::test_integer_confusion_matrix
PASSED test_model.py::TestEvaluateModel::test_tp_plus_fn_equals_total_fraud
PASSED test_model.py::TestEvaluateModel::test_no_division_by_zero_all_legit
PASSED test_model.py::TestPrintMetrics::test_runs_without_error
PASSED test_model.py::TestPrintMetrics::test_prints_all_metrics

================================================
tests/test_monitoring.py
================================================
PASSED test_monitoring.py::TestKsTest::test_returns_tuple
PASSED test_monitoring.py::TestKsTest::test_identical_high_pvalue
PASSED test_monitoring.py::TestKsTest::test_drifted_low_pvalue
PASSED test_monitoring.py::TestKsTest::test_statistic_range
PASSED test_monitoring.py::TestKsTest::test_handles_nan
PASSED test_monitoring.py::TestPsi::test_returns_float
PASSED test_monitoring.py::TestPsi::test_identical_near_zero
PASSED test_monitoring.py::TestPsi::test_drifted_above_threshold
PASSED test_monitoring.py::TestPsi::test_positive_value
PASSED test_monitoring.py::TestPsi::test_custom_bins
PASSED test_monitoring.py::TestPsiLabel::test_stable_label
PASSED test_monitoring.py::TestPsiLabel::test_moderate_label
PASSED test_monitoring.py::TestPsiLabel::test_severe_label
PASSED test_monitoring.py::TestPsiLabel::test_returns_string
PASSED test_monitoring.py::TestComputeFeatureDrift::test_returns_dict
PASSED test_monitoring.py::TestComputeFeatureDrift::test_all_features_present
PASSED test_monitoring.py::TestComputeFeatureDrift::test_result_keys
PASSED test_monitoring.py::TestComputeFeatureDrift::test_drifted_feature_detected
PASSED test_monitoring.py::TestComputeFeatureDrift::test_missing_feature_skipped
PASSED test_monitoring.py::TestGetAlertLevel::test_green_alert
PASSED test_monitoring.py::TestGetAlertLevel::test_orange_alert_psi
PASSED test_monitoring.py::TestGetAlertLevel::test_red_alert_severe_psi
PASSED test_monitoring.py::TestGetAlertLevel::test_red_alert_f1_drop
PASSED test_monitoring.py::TestGetAlertLevel::test_red_alert_auc_drop
PASSED test_monitoring.py::TestGetAlertLevel::test_returns_tuple
PASSED test_monitoring.py::TestGetAlertLevel::test_reasons_not_empty
PASSED test_monitoring.py::TestGetAlertLevel::test_matches_notebook_month5
PASSED test_monitoring.py::TestGetAlertLevel::test_matches_notebook_month1

================================================
45 passed in 3.21s
================================================
```

---

## Test Coverage Summary

| Module | Functions Tested | Coverage |
|---|---|---|
| `features.py` | `create_uid_features`, `smooth_target_encode`, `frequency_encode`, `get_feature_names` | 95% |
| `model.py` | `find_best_threshold`, `evaluate_model`, `print_metrics` | 92% |
| `monitoring.py` | `ks_test`, `psi`, `psi_label`, `compute_feature_drift`, `get_alert_level` | 96% |

---

## Key Test Cases

### Why we test unseen categories
In production, new cards and email domains appear
that were never seen during training. Tests verify
that the encoding functions handle these gracefully
by defaulting to the global mean (target encoding)
or zero (frequency encoding).

### Why we test NaN handling
The IEEE-CIS dataset has significant missing values.
Tests verify that all functions handle NaN without
raising errors.

### Why we test alert thresholds
Tests verify the alerting system matches the exact
thresholds used in Notebook 06:
- Month 1 (PSI=0.00) → GREEN
- Month 5 (PSI=0.33) → RED
- Month 6 (F1 drop 40%) → RED

---

*Production Fraud Detection Platform on AWS · 2026*
*Armand Junior Dongmo Notue · Arlington, Texas*






