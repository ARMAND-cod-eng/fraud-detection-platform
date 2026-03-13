# pipelines/ Real-Time Streaming Pipelines

This folder contains the two real-time fraud detection
pipelines built on Amazon Kinesis, as demonstrated
in Notebook 04.

---

## Files

| File | Approach | Dataset | F1 |
|---|---|---|---|
| `paysim_producer.py` | Domain Rules | PaySim synthetic | 0.923 |
| `ieee_producer.py` | V3 XGBoost | IEEE-CIS real-world | 0.919 |

---

## Architecture
```
Transaction arrives
        │
        ▼
┌───────────────────┐
│  Kinesis Stream   │  fraud-detection-stream
│  1 shard          │  $0.015/hour
│  24hr retention   │
└────────┬──────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
PaySim      IEEE-CIS
Rules       V3 XGBoost
Pipeline    Pipeline
    │         │
    ▼         ▼
F1: 0.923  F1: 0.919
Rec: 1.000 Prec: 1.000
```

---

## Pipeline A : PaySim Domain Rules

### How it works
Applies 5 hand-crafted business rules to compute
a fraud score for each transaction. No ML model needed.

### Rules

| Rule | Score | Rationale |
|---|---|---|
| Wrong transaction type | +0.01 | Only TRANSFER/CASH_OUT carry fraud |
| Balance wiped to zero | +0.50 | Account fully drained |
| Large amount > $200,000 | +0.20 | Unusually high amount |
| Destination unchanged | +0.30 | Money did not arrive |
| Exact drain | +0.20 | Amount equals origin balance |

### Results

| Metric | Value |
|---|---|
| Precision | 0.857 |
| Recall | 1.000 |
| F1-Score | 0.923 |
| False Alarms | 1 (large legitimate payment) |

### Usage
```python
from paysim_producer import run_paysim_pipeline

results = run_paysim_pipeline(n_transactions=100)
```

---

## Pipeline B : IEEE-CIS V3 XGBoost

### How it works
Scores each transaction using the V3 XGBoost model
(626 features, AUC 0.9622) with a decision threshold
of 0.87.

### Results

| Metric | Value |
|---|---|
| Precision | 1.000 |
| Recall | 0.850 |
| F1-Score | 0.919 |
| False Alarms | 0 |

### Usage
```python
from ieee_producer import run_ieee_pipeline

results = run_ieee_pipeline()
```

---

## Prerequisites

### 1. Kinesis stream must be running
```python
from infrastructure.setup_sagemaker import (
    create_kinesis_stream
)
create_kinesis_stream('fraud-detection-stream')
```

### 2. V3 model must be in S3
```
s3://fraud-detection-mlproject-armand/
└── models/v3/
    ├── xgb_model_v3fix.json
    └── feature_names_v3fix.json
```

### 3. df_features_v3.csv must be in S3
```
s3://fraud-detection-mlproject-armand/
└── processed-data/
    └── df_features_v3.csv
```

---

## Important Kinesis Iterator Expiry

Kinesis LATEST iterators expire after **5 minutes**.
Both pipelines get the iterator immediately before
consuming — do not add delays between producer and
consumer steps.
```python
# CORRECT — iterator created right before use
iterator = get_latest_iterator(kin, stream_name)
results  = consume_and_score(kin, stream_name,
                              iterator, ...)

# WRONG — iterator will expire during download
s3.download_file(...)  # takes 6 minutes!
iterator = get_latest_iterator(...)  # too late!
```

---

## Side-by-Side Comparison

| Dimension | PaySim Rules | IEEE-CIS V3 |
|---|---|---|
| Precision | 0.857 | **1.000** |
| Recall | **1.000** | 0.850 |
| F1-Score | **0.923** | 0.919 |
| False Alarms | 1 | **0** |
| Features needed | 5 rules | 626 features |
| Training required | No | Yes |
| Adapts to new fraud | No | Yes |
| Explainable | Yes | Partially |

### When to use each

**Domain Rules**: best when:
- New product with no fraud history
- No labeled training data available
- Regulators require full explainability
- Rapid deployment needed

**V3 XGBoost**: best when:
- Large labeled dataset available
- Complex evolving fraud patterns
- Customer experience is priority (zero false alarms)
- Long-term production pipeline

**Production recommendation, use both:**
```
Layer 1: Rules    → block obvious fraud instantly
Layer 2: V3 Model → score everything else deeply
```

---

## Cost Management
```python
# Always delete stream after demo!
from infrastructure.setup_sagemaker import (
    delete_kinesis_stream
)
delete_kinesis_stream('fraud-detection-stream')
# Saving $0.015/hour!
```

---

*Production Fraud Detection Platform on AWS · 2026*
*Armand Junior Dongmo Notue · Arlington, Texas*




