# **src/: Production Source Modules**

This folder contains the core Python modules extracted
from the 6 notebooks and refactored into reusable,
production-ready code.

---

## **Module Overview**

| Module | Purpose |
|---|---|
| `features.py` | Feature engineering functions |
| `model.py` | Model training, scoring, evaluation |
| `streaming.py` | Kinesis producer + consumer |
| `monitoring.py` | Drift detection + alerting |
| `__init__.py` | Package exports |

---

## **features.py**

Feature engineering pipeline used to transform raw
IEEE-CIS transaction data into the 626-feature matrix
used by the V3 XGBoost model.

### **Functions**

#### **`create_uid_features(df)`**
Creates 3 composite identity keys from card and address columns:

```python
from src.features import create_uid_features

df = create_uid_features(df)
# Adds: uid1 = card1_addr1
#       uid2 = card1_addr1_P_emaildomain
#       uid3 = card1_card2
```

#### **`smooth_target_encode(train, val, col, target, global_mean, k=20)`**
Smoothed Bayesian target encoding that prevents overfitting on rare categories.

```python
from src.features import smooth_target_encode

global_mean = train['isFraud'].mean()
train_enc, val_enc = smooth_target_encode(
    train, val,
    col='card1',
    target='isFraud',
    global_mean=global_mean,
    k=20
)
train['card1_fraud_rate'] = train_enc
val['card1_fraud_rate']   = val_enc
```

**Why smoothing matters:**
Standard target encoding assigns the raw fraud rate
to each category. A card seen only once gets a fraud
rate of either 0.0 or 1.0 — unreliable. Smoothing
pulls rare categories toward the global mean:

```
encoded = (count × mean + k × global_mean) / (count + k)
```

With k=20: a card seen 5 times gets ~80% pulled toward
the global mean. A card seen 200 times uses mostly its
own observed rate.

#### **`frequency_encode(train, val, cols)`**
Replaces each category with its relative frequency in the training set.

```python
from src.features import frequency_encode

train, val = frequency_encode(
    train, val,
    cols=['card1', 'card2', 'addr1', 'P_emaildomain']
)
# Adds: card1_freq, card2_freq, addr1_freq, P_emaildomain_freq
```

#### **`add_target_encoding_features(train, val, target, k=20)`**
Applies smoothed target encoding to all UID and card/email columns in one call.

```python
from src.features import add_target_encoding_features

train, val = add_target_encoding_features(
    train, val,
    target='isFraud',
    k=20
)
# Adds 16 features:
# uid1_fraud_rate, uid1_count
# uid2_fraud_rate, uid2_count
# uid3_fraud_rate, uid3_count
# card1_fraud_rate, card1_count
# card2_fraud_rate, card2_count
# addr1_fraud_rate, addr1_count
# P_emaildomain_fraud_rate, P_emaildomain_count
# R_emaildomain_fraud_rate, R_emaildomain_count
```

#### **`get_feature_names(df)`**
Returns the final list of numeric feature columns, excluding target and string columns.

```python
from src.features import get_feature_names

feature_names = get_feature_names(df)
# Returns list of 626 feature names
```

---

## **model.py**

Scoring and evaluation utilities for the V3 XGBoost
fraud detection model.

### **Functions**

#### **`score_batch(batch, model, feature_names)`**
Score a batch of transactions using the loaded XGBoost model.

```python
import xgboost as xgb
from src.model import score_batch

model = xgb.Booster()
model.load_model('models/v3/xgb_model_v3fix.json')

scores = score_batch(batch_df, model, feature_names)
# Returns np.ndarray of fraud probabilities (0 to 1)
```

#### **`find_best_threshold(y_true, y_scores)`**
Find the decision threshold that maximises F1-Score on a validation set.

```python
from src.model import find_best_threshold

threshold, f1 = find_best_threshold(y_val, val_scores)
# Returns (0.87, 0.7545) for V3 model
```

#### **`evaluate_model(y_true, y_scores, threshold=0.87)`**
Compute full evaluation metrics including confusion matrix.

```python
from src.model import evaluate_model, print_metrics

metrics = evaluate_model(y_val, val_scores, threshold=0.87)
print_metrics(metrics)

# Output:
#    AUC-ROC   : 0.9622
#    AUC-PR    : 0.7968
#    F1-Score  : 0.7545
#    Precision : 0.8196
#    Recall    : 0.6990
```

---

## **streaming.py**

Kinesis stream producer and consumer for real-time
fraud scoring.

### **Functions**

#### **`send_transactions(kin, stream_name, transactions, feature_names)`**
Send a batch of transactions to a Kinesis stream.

```python
import boto3
from src.streaming import get_kinesis_client, send_transactions

kin    = get_kinesis_client(region='us-east-1')
result = send_transactions(
    kin,
    stream_name   = 'fraud-detection-stream',
    transactions  = df_batch,
    feature_names = feature_names
)
# result = {sent_count: 100, throughput: 71 records/sec}
```

#### **`consume_and_score(kin, stream_name, iterator, model, ...)`**
Consume records from Kinesis and score each transaction.

```python
from src.streaming import get_latest_iterator, consume_and_score

iterator = get_latest_iterator(kin, 'fraud-detection-stream')
results  = consume_and_score(
    kin, 'fraud-detection-stream',
    iterator, model, feature_names,
    threshold=0.87
)
# Returns list of scored transaction dicts
```

#### **`delete_stream(kin, stream_name)`**
Delete the Kinesis stream to stop incurring costs.

```python
from src.streaming import delete_stream

delete_stream(kin, 'fraud-detection-stream')
# Stream deleted: fraud-detection-stream
# Cost savings: $0.015/hour
```

---

## **monitoring.py**

Feature drift detection and model performance alerting
using KS Test and Population Stability Index (PSI).

### **Functions**

#### **`ks_test(baseline, current)`**
Kolmogorov-Smirnov test for statistical distribution drift.

```python
from src.monitoring import ks_test

ks_stat, p_value = ks_test(
    baseline['card1_fraud_rate'],
    current['card1_fraud_rate']
)
# p_value < 0.05 = significant drift detected
```

#### **`psi(baseline, current, bins=10)`**
Population Stability Index — industry standard drift metric.

```python
from src.monitoring import psi, psi_label

psi_score = psi(
    baseline['P_emaildomain_fraud_rate'],
    current['P_emaildomain_fraud_rate']
)
label = psi_label(psi_score)
# psi_score = 0.33 -> label = 'SEVERE'
```

| PSI Score | Label | Action |
|---|---|---|
| < 0.10 | Stable | No action |
| 0.10 – 0.25 | Moderate | Monitor closely |
| > 0.25 | SEVERE | Retrain model! |

#### **`get_alert_level(auc, f1, max_psi, base_auc, base_f1)`**
Determine monitoring alert level.

```python
from src.monitoring import get_alert_level, print_alert

level, reasons = get_alert_level(
    auc=0.9058, f1=0.5058, max_psi=0.16,
    base_auc=0.9976, base_f1=0.8363
)
print_alert('Month 6', level, reasons)

# [ALERT] Month 6 — RED
#    -> F1=0.5058 dropped 40% from baseline
```

---

## **Installation**

```bash
pip install -r requirements.txt
```

## **Usage Example: Full Pipeline**

```python
import xgboost as xgb
import boto3
import json
import pandas as pd

from src.features  import create_uid_features, add_target_encoding_features
from src.model     import score_batch, evaluate_model, print_metrics
from src.streaming import get_kinesis_client, get_latest_iterator
from src.streaming import send_transactions, consume_and_score
from src.monitoring import compute_feature_drift, get_alert_level

# 1. Load model
model = xgb.Booster()
model.load_model('s3://fraud-detection-mlproject-armand/models/v3/xgb_model_v3fix.json')

with open('models/v3/feature_names_v3fix.json') as f:
    feature_names = json.load(f)

# 2. Score a batch
scores  = score_batch(batch_df, model, feature_names)
metrics = evaluate_model(batch_df['isFraud'], scores)
print_metrics(metrics)

# 3. Stream transactions
kin      = get_kinesis_client()
iterator = get_latest_iterator(kin, 'fraud-detection-stream')
results  = consume_and_score(kin, 'fraud-detection-stream',
                              iterator, model, feature_names)

# 4. Monitor drift
drift   = compute_feature_drift(baseline_df, current_df, feature_names)
max_psi = max(v['psi'] for v in drift.values())
level, reasons = get_alert_level(
    metrics['auc_roc'], metrics['f1'], max_psi,
    base_auc=0.9976, base_f1=0.8363
)
```

---

*Production Fraud Detection Platform on AWS · 2026*
*Armand Junior Dongmo Notue · Arlington, Texas*
