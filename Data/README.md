# data/ : Datasets

This folder documents the three datasets used in the
Production Fraud Detection Platform. Raw data files
are **not stored in this repository** (files are too
large for GitHub). All datasets are downloaded from
Kaggle and stored in the AWS S3 data lake.

---

## Datasets Overview

| Dataset | Transactions | Fraud Rate | Size | Notebook |
|---|---|---|---|---|
| IEEE-CIS Fraud Detection | 590,540 | 3.5% | ~710 MB | 01, 02, 03, 04 |
| PaySim Synthetic | 6,362 | 7.5% | ~1 MB | 04 |
| ULB Credit Card Fraud | 284,807 | 0.17% | ~144 MB | 05 |

---

## Dataset 1 — IEEE-CIS Fraud Detection

### Source
- **Platform:** Kaggle
- **URL:** https://www.kaggle.com/competitions/ieee-fraud-detection/data
- **Hosted by:** IEEE Computational Intelligence Society (IEEE-CIS)
- **Competition:** IEEE-CIS Fraud Detection (2019)

### Files

| File | Size | Rows | Description |
|---|---|---|---|
| `train_transaction.csv` | 683 MB | 590,540 | Transaction features |
| `train_identity.csv` | 26 MB | 144,233 | Identity features |

### Features

| Group | Count | Description |
|---|---|---|
| Transaction info | 6 | TransactionDT, TransactionAmt, ProductCD, dist1, dist2 |
| Card features | 6 | card1–card6 (card type, bank, country) |
| Address | 2 | addr1, addr2 |
| Email domain | 2 | P_emaildomain, R_emaildomain |
| Count features | 14 | C1–C14 (counting variables) |
| Time delta | 9 | D1–D15 (time delta features) |
| Match features | 11 | M1–M9 (match features) |
| V-features | 339 | V1–V339 (Vesta engineered features) |
| Identity | 41 | DeviceType, DeviceInfo, id_01–id_38 |

### Class Distribution

```
Legitimate : 569,877 (96.5%)
Fraud      :  20,663  (3.5%)
Imbalance  : 28:1
```

### S3 Location
```
s3://fraud-detection-mlproject-armand/raw-data/train_transaction.csv
s3://fraud-detection-mlproject-armand/raw-data/train_identity.csv
```

### How to Download
```python
import kaggle

kaggle.api.competition_download_files(
    'ieee-fraud-detection',
    path='/tmp/'
)
```

Or via CLI:
```bash
kaggle competitions download \
    -c ieee-fraud-detection \
    -p /tmp/
```

### Usage in this project
- **Notebook 01:** EDA + feature engineering → 626 features
- **Notebook 02:** Model training V1→V4 → V3 AUC 0.9622
- **Notebook 03:** SageMaker deployment → 36.9ms latency
- **Notebook 04:** Real-time Kinesis pipeline → F1 0.919

---

## Dataset 2 — PaySim Synthetic Financial Transactions

### Source
- **Platform:** Kaggle
- **URL:** https://www.kaggle.com/datasets/ealaxi/paysim1
- **Author:** Edgar Alonso Lopez-Rojas
- **Paper:** PaySim: A financial mobile money simulator for fraud detection

### Files

| File | Size | Rows | Description |
|---|---|---|---|
| `PS_20174392719_1491204439457_log.csv` | ~470 MB | 6,362,620 | Full simulation |

> Note: We use a 6,362 row sample (0.1%) for the streaming demo.

### Features

| Feature | Description |
|---|---|
| `step` | Time step (1 step = 1 hour) |
| `type` | Transaction type (PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN) |
| `amount` | Transaction amount |
| `nameOrig` | Origin account |
| `oldbalanceOrg` | Origin balance before |
| `newbalanceOrig` | Origin balance after |
| `nameDest` | Destination account |
| `oldbalanceDest` | Destination balance before |
| `newbalanceDest` | Destination balance after |
| `isFraud` | Fraud label (0/1) |
| `isFlaggedFraud` | System flag (0/1) |

### Class Distribution

```
Legitimate : 5,896 (92.7%)
Fraud      :   466  (7.3%)
Imbalance  : 13:1
```

### Fraud Patterns in PaySim

Fraud only occurs in two transaction types:
```
TRANSFER  : Fraudulent transfers to mule accounts
CASH_OUT  : Fraudulent cash withdrawals

Signatures:
   - Origin balance wiped to exactly 0
   - Destination balance unchanged
   - Amount equals origin balance (exact drain)
```

### S3 Location
```
s3://fraud-detection-mlproject-armand/raw-data/paysim_sample.csv
```

### How to Download
```bash
kaggle datasets download \
    -d ealaxi/paysim1 \
    -p /tmp/
```

### Usage in this project
- **Notebook 04:** Domain rules streaming pipeline
  - 5 business rules applied
  - Precision 0.857, Recall 1.000, F1 0.923
  - Sent through Amazon Kinesis stream

---

## Dataset 3 — ULB Credit Card Fraud Detection

### Source
- **Platform:** Kaggle
- **URL:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- **Author:** Machine Learning Group, Université Libre de Bruxelles (ULB)
- **Paper:** Calibrated Probability Estimation for Fraud Detection (Dal Pozzolo et al., 2015)

### Files

| File | Size | Rows | Description |
|---|---|---|---|
| `creditcard.csv` | 143.8 MB | 284,807 | European cardholders (Sept 2013) |

### Features

| Feature | Description |
|---|---|
| `V1`–`V28` | PCA components (anonymized for privacy) |
| `Amount` | Transaction amount in EUR |
| `Time` | Seconds elapsed since first transaction |
| `Class` | Fraud label (0=legitimate, 1=fraud) |

> **Note:** Raw features are not available. PCA transformation
> was applied by ULB to protect cardholder identity.
> This means no domain feature engineering is possible —
> making unsupervised anomaly detection the right approach.

### Class Distribution

```
Legitimate : 284,315 (99.827%)
Fraud      :     492  (0.173%)
Imbalance  : 578:1
```

This is the most extreme class imbalance of all
three datasets — 20x more imbalanced than IEEE-CIS.

### Time Coverage
```
Duration : 48 hours
Timezone : UTC+1 (Central European Time)
Region   : European cardholders
```

### Amount Statistics

| Metric | Legitimate | Fraud |
|---|---|---|
| Mean | $88.29 | $122.21 |
| Median | $22.00 | $9.25 |
| Max | $25,691.16 | $2,125.87 |

Fraud median ($9.25) is much lower than legitimate ($22.00)
— consistent with fraudsters testing stolen cards with
small amounts before making larger purchases.

### S3 Location
```
s3://fraud-detection-mlproject-armand/raw-data/creditcard.csv
```

### How to Download
```bash
kaggle datasets download \
    -d mlg-ulb/creditcardfraud \
    -p /tmp/
```

### Usage in this project
- **Notebook 05:** Unsupervised anomaly detection
  - Isolation Forest: AUC 0.9473, F1 0.2169
  - Autoencoder: AUC 0.9420, F1 0.4140 **(best)**
  - One-Class SVM: AUC 0.9490, F1 0.3806
  - Key finding: 45.9x reconstruction error separation

---

## Processed Data

In addition to raw datasets, the following processed
files are stored in S3 after running the notebooks:

| File | Size | Features | Description |
|---|---|---|---|
| `df_features_v2.csv` | ~800 MB | 621 | IEEE-CIS V2 features |
| `df_features_v3.csv` | 1.3 GB | 626 | IEEE-CIS V3 features (PRODUCTION) |

### S3 Location
```
s3://fraud-detection-mlproject-armand/processed-data/df_features_v2.csv
s3://fraud-detection-mlproject-armand/processed-data/df_features_v3.csv
```

---

## How to Upload Datasets to S3

### Option A — AWS Console
```
S3 → fraud-detection-mlproject-armand
→ raw-data/ → Upload → select file
```

### Option B — AWS CLI
```bash
aws s3 cp train_transaction.csv \
    s3://fraud-detection-mlproject-armand/raw-data/ \
    --profile ml-engineer

aws s3 cp train_identity.csv \
    s3://fraud-detection-mlproject-armand/raw-data/ \
    --profile ml-engineer

aws s3 cp creditcard.csv \
    s3://fraud-detection-mlproject-armand/raw-data/ \
    --profile ml-engineer
```

### Option C — Kaggle API directly from SageMaker
```bash
# Install Kaggle CLI
pip install kaggle

# Set credentials
mkdir -p ~/.kaggle
echo '{"username":"YOUR_USERNAME","key":"YOUR_KEY"}' \
    > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Download IEEE-CIS
kaggle competitions download \
    -c ieee-fraud-detection -p /tmp/

# Download ULB
kaggle datasets download \
    -d mlg-ulb/creditcardfraud -p /tmp/

# Upload to S3
aws s3 cp /tmp/train_transaction.csv \
    s3://fraud-detection-mlproject-armand/raw-data/
aws s3 cp /tmp/creditcard.csv \
    s3://fraud-detection-mlproject-armand/raw-data/
```

---

## Dataset Comparison

| Property | IEEE-CIS | PaySim | ULB |
|---|---|---|---|
| **Type** | Real-world | Synthetic | Real-world |
| **Transactions** | 590,540 | 6,362 | 284,807 |
| **Fraud rate** | 3.5% | 7.5% | 0.17% |
| **Imbalance** | 28:1 | 13:1 | 578:1 |
| **Features** | 434 raw → 626 engineered | 10 | 30 (PCA) |
| **Labels available** | Yes | Yes | Yes |
| **Raw features** | Yes | Yes | No (PCA only) |
| **ML approach** | Supervised | Rules | Unsupervised |
| **Best model** | XGBoost V3 | Domain Rules | Autoencoder |
| **Best AUC** | 0.9622 | N/A | 0.9490 |
| **Best F1** | 0.7545 | 0.923 | 0.4140 |

---

*Production Fraud Detection Platform on AWS · 2026*
*Armand Junior Dongmo Notue · Arlington, Texas*
