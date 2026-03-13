# **Production Fraud Detection Platform on AWS**

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1.4-orange)
![AWS](https://img.shields.io/badge/AWS-SageMaker%20%7C%20Kinesis%20%7C%20S3-yellow)
![License](https://img.shields.io/badge/License-MIT-green)
![Tests](https://img.shields.io/badge/Tests-45%20passed-brightgreen)

</div>

> End-to-end ML platform covering data engineering, supervised model training,
> real-time deployment, streaming pipelines, unsupervised anomaly detection,
> and production monitoring — built entirely on AWS.

**Author:** Armand Junior Dongmo Notue · Arlington, Texas
**Stack:** Python · XGBoost · LightGBM · CatBoost · AWS SageMaker · Amazon Kinesis · S3 · IAM

---

## **Table of Contents**

- [Project Overview](#project-overview)
- [Results Summary](#results-summary)
- [Architecture](#architecture)
- [Datasets](#datasets)
- [Model Progression](#model-progression)
- [Notebook Details](#notebook-details)
- [Repository Structure](#repository-structure)
- [Technology Stack](#technology-stack)
- [Key Technical Decisions](#key-technical-decisions)
- [AWS Infrastructure](#aws-infrastructure)
- [How to Run](#how-to-run)
- [Unit Tests](#unit-tests)
- [Contact](#contact)

---

## **Project Overview**

Most ML portfolios stop at model training. This project covers the **complete
production ML lifecycle** — from raw CSV files to a deployed real-time endpoint
with drift monitoring and automated alerting.

The platform is built across **6 notebooks**, using **3 real-world datasets**
and covering every stage a fraud detection system goes through in production:

```
Raw Data → Feature Engineering → Model Training → Deployment
→ Real-Time Streaming → Anomaly Detection → Monitoring + Alerting
```

---

## **Results Summary**

| Notebook | Topic | Key Result |
|---|---|---|
| 01 | Data Ingestion + EDA | 590,540 transactions loaded, 626 features engineered |
| 02 | Model Training V1→V4 | V3 XGBoost **AUC 0.9622**, F1 0.7545 |
| 03 | SageMaker Deployment | Real-time endpoint, avg **36.9ms latency** |
| 04 | Real-Time Streaming | Kinesis pipeline **F1 0.919**, zero false alarms |
| 05 | Anomaly Detection | Autoencoder **AUC 0.942**, 45.9x separation ratio |
| 06 | Model Monitoring | Drift detected Month 5, **PSI 0.33 RED alert** |

---

## **Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   Raw Data (S3)                                             │
│       │                                                     │
│       ▼                                                     │
│   Feature Engineering (Notebook 01)                        │
│       434 base features → 626 with target encoding         │
│       │                                                     │
│       ▼                                                     │
│   Model Training (Notebook 02)                             │
│       V1 → V2 → V3 (PRODUCTION) → V4                       │
│       XGBoost + LightGBM + CatBoost                        │
│       │                                                     │
│       ▼                                                     │
│   SageMaker Deployment (Notebook 03)                       │
│       Real-time endpoint · ml.m5.xlarge · 36.9ms           │
│       │                                                     │
│       ├─────────────────────────┐                          │
│       ▼                         ▼                          │
│   Streaming (Notebook 04)   Anomaly Detection (Notebook 05)│
│   Amazon Kinesis             ULB dataset                   │
│   PaySim Rules F1 0.923      Isolation Forest              │
│   IEEE-CIS V3   F1 0.919     Autoencoder                   │
│                              One-Class SVM                 │
│       │                         │                          │
│       └────────────┬────────────┘                          │
│                    ▼                                        │
│           Model Monitoring (Notebook 06)                   │
│           KS Test + PSI drift detection                    │
│           GREEN / ORANGE / RED alerting                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## **Datasets**

| Dataset | Source | Transactions | Fraud Rate | Imbalance | Notebook |
|---|---|---|---|---|---|
| [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/data) | Kaggle | 590,540 | 3.5% | 28:1 | 01–04 |
| [PaySim Synthetic](https://www.kaggle.com/datasets/ealaxi/paysim1) | Kaggle | 6,362 | 7.5% | 13:1 | 04 |
| [ULB Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | Kaggle | 284,807 | 0.17% | 578:1 | 05 |

**Why three datasets?**

Each dataset represents a different fraud detection challenge:
- **IEEE-CIS** — real-world complexity with 434 raw features and rich identity data
- **PaySim** — synthetic data ideal for demonstrating rule-based detection
- **ULB** — extreme imbalance (578:1) with anonymized PCA features — perfect for unsupervised detection

---

## **Model Progression**

| Version | AUC-ROC | AUC-PR | F1 | Features | Notes |
|---|---|---|---|---|---|
| V1 | 0.9533 | 0.6993 | 0.4288 | 142 | Baseline XGBoost + LightGBM |
| V2 | 0.9589 | 0.7361 | 0.4817 | 621 | + V1–V339 identity features |
| **V3** | **0.9622** | **0.7984** | **0.7545** | **626** | **+ Smoothed target encoding → PRODUCTION** |
| V4 | 0.9583 | 0.7737 | 0.7306 | 626 | + CatBoost stacking ensemble |

**Why V3 and not V4?**

V4 adds CatBoost to create a 3-model stacking ensemble. AUC improves
marginally (0.9627) but F1 drops from 0.7545 to 0.7306. The added complexity
is not justified by the result. V3 is the production model.

**The V1→V3 breakthrough:**

The F1 jump from 0.4288 to 0.7545 (+75%) came entirely from adding smoothed
Bayesian target encoding on UID composite keys:

```python
uid1 = card1 + addr1
uid2 = card1 + addr1 + P_emaildomain
uid3 = card1 + card2

# Smoothed encoding (k=20):
encoded = (count × fraud_rate + 20 × global_mean) / (count + 20)
```

This captures historical fraud rates per card/address combination while
preventing overfitting on rare categories — the single most impactful
feature engineering decision in this project.

---

## **Notebook Details**

### Notebook 01 — Data Ingestion + EDA
- Loads IEEE-CIS datasets from S3 (683MB + 26MB)
- Merges transaction and identity tables on TransactionID
- Handles missing values (45% in V-features, 30% in identity)
- Engineers 434 base features + 192 V-features
- EDA: class imbalance, amount distributions, time patterns

### Notebook 02 — Model Training
- 4 model versions with documented rationale for each iteration
- V3 uses 700 XGBoost estimators + 700 LightGBM estimators
- Optimal threshold search: 0.87 maximises F1 on 118,108 validation rows
- Scale_pos_weight = 27.6 to handle 28:1 class imbalance

```
V3 Ensemble (threshold=0.87):
   AUC-ROC   : 0.9627
   AUC-PR    : 0.7984
   F1-Score  : 0.7545
   Precision : 0.8196
   Recall    : 0.6990
   Fraud caught : 2,889 / 4,133 (69.9%)
   False alarms : 636 / 114,607 (0.56%)
```

### Notebook 03 — SageMaker Deployment
- Packages V3 XGBoost as AWS built-in container (xgboost:1.7-1)
- Deploys to ml.m5.xlarge real-time endpoint
- Latency test across 100 live requests:

```
Min latency : 18ms
Avg latency : 36.9ms
P99 latency : 131.7ms
```

- Endpoint deleted after testing (zero ongoing cost)
- Documents failed ensemble deployment attempts — shows scientific thinking

### Notebook 04 — Real-Time Streaming
- Creates Amazon Kinesis stream (1 shard, us-east-1)
- **Pipeline A — PaySim Domain Rules:**
  - 5 hand-crafted business rules
  - Precision 0.857 · Recall 1.000 · F1 0.923
- **Pipeline B — IEEE-CIS V3 XGBoost:**
  - 626-feature real-time scoring
  - Precision 1.000 · Recall 0.850 · F1 0.919
  - Zero false positives — model never wrongly blocks a card

### Notebook 05 — Anomaly Detection
- Dataset: ULB (284,807 transactions, 0.17% fraud)
- All 3 models trained on **legitimate transactions only** — no fraud labels used

```
Method             AUC-ROC   AUC-PR   F1     Fraud Caught  Train Time
Isolation Forest   0.9473    0.1106   0.2169  109/492       3.8s
Autoencoder        0.9420    0.3068   0.4140  267/492       26.7s
One-Class SVM      0.9490    0.2560   0.3806  278/492       0.4s
```

Key finding: Autoencoder reconstruction error is **45.9x higher** for fraud
transactions than legitimate ones — the strongest unsupervised signal in the project.

### Notebook 06 — Model Monitoring
- Simulates 6 months of production traffic (50,000 transactions)
- Injects 4 realistic drift scenarios:

| Month | Scenario | Key Change |
|---|---|---|
| 1–2 | Baseline | No drift |
| 3 | Small-amount fraud | TransactionAmt shifts to $1–$15 |
| 4 | Card network compromise | card1_fraud_rate ×4 |
| 5 | New fraud domains | P_emaildomain_fraud_rate → 0.33 PSI |
| 6 | Full concept drift | All signals weakened ×0.3 |

Alert timeline:
```
Month 1  GREEN    Baseline — AUC 0.9976, F1 0.8363
Month 2  ORANGE   PSI=0.11 moderate drift
Month 3  ORANGE   PSI=0.19 moderate drift
Month 4  ORANGE   PSI=0.16 moderate drift
Month 5  RED      PSI=0.33 SEVERE → Retrain immediately!
Month 6  RED      F1 dropped 40% from baseline
```

---

## **Repository Structure**

```
fraud-detection-platform/
│
├── data/
│   └── README.md              Dataset documentation + Kaggle links
│
├── dashboard/
│   └── README.md              Chart index from all 6 notebooks
│
├── infrastructure/
│   ├── setup_aws.sh           S3 + IAM + Kinesis setup
│   ├── setup_sagemaker.py     SageMaker utilities
│   └── README.md
│
├── notebooks/
│   ├── 01_data_ingestion_eda.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_deployment.ipynb
│   ├── 04_streaming.ipynb
│   ├── 05_anomaly_detection.ipynb
│   └── 06_monitoring.ipynb
│
├── pipelines/
│   ├── paysim_producer.py     PaySim domain rules pipeline
│   ├── ieee_producer.py       IEEE-CIS V3 XGBoost pipeline
│   └── README.md
│
├── src/
│   ├── __init__.py
│   ├── features.py            Feature engineering functions
│   ├── model.py               Scoring + evaluation
│   ├── streaming.py           Kinesis producer + consumer
│   ├── monitoring.py          Drift detection + alerting
│   └── README.md
│
├── tests/
│   ├── test_features.py       14 unit tests
│   ├── test_model.py          17 unit tests
│   ├── test_monitoring.py     18 unit tests
│   └── README.md
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## **Technology Stack**

### Machine Learning
| Library | Version | Usage |
|---|---|---|
| XGBoost | 2.1.4 | Primary production model (V3) |
| LightGBM | 4.0.0 | Ensemble component |
| CatBoost | 1.2.0 | V4 ensemble experiment |
| Scikit-learn | 1.7.2 | Isolation Forest, OC-SVM, metrics |

### Data
| Library | Version | Usage |
|---|---|---|
| Pandas | 2.3.3 | Data manipulation |
| NumPy | 1.26.4 | Numerical computing |
| SciPy | 1.13.0 | KS test for drift detection |

### AWS
| Service | Usage |
|---|---|
| S3 | Data lake (raw data, features, models) |
| SageMaker | JupyterLab + model training + endpoint |
| Kinesis | Real-time transaction streaming |
| IAM | User and role management |

### Visualization
| Library | Version | Usage |
|---|---|---|
| Matplotlib | 3.10.8 | All charts and dashboards |
| Seaborn | 0.13.2 | Statistical visualizations |

---

## **Key Technical Decisions**

### 1. Smoothed Target Encoding (k=20)
Standard target encoding assigns the raw fraud rate to each category,
causing overfitting on rare cards or email domains. With k=20:

```
encoded = (count × fraud_rate + 20 × global_mean) / (count + 20)
```

A card seen 5 times gets 80% pulled toward the global mean.
A card seen 200 times uses mostly its own observed rate.
This single decision drove F1 from 0.48 → 0.75.

### 2. Threshold 0.87
In fraud detection, false positives (blocking legitimate customers)
are more damaging than false negatives (missing some fraud). The 0.87
threshold maximises F1 while maintaining Precision > 0.80 and
keeping the false alarm rate below 0.6%.

### 3. Feature Store Requirement
The V3 model's target-encoded features must be pre-computed on the
full training history (590,540 rows). When computed on only 10,000 rows,
recall drops from 0.85 to 0.10. In production, these features are
served from a feature store (AWS SageMaker Feature Store or Redis)
updated on a daily batch schedule.

### 4. Two-Layer Production Architecture
```
Layer 1: Domain Rules  → block obvious fraud instantly
                         (balance wiped, wrong type)
Layer 2: V3 XGBoost    → score everything else deeply
                         (subtle behavioral patterns)
```
This is the standard architecture at Capital One, JPMorgan, and PayPal.

### 5. PSI + KS Test for Drift Detection
PSI alone can miss distributional shifts that affect model performance
without changing the marginal distribution. Using both PSI (distribution
shift) and KS test (statistical significance) provides complementary
coverage:

| Test | Detects | Threshold |
|---|---|---|
| KS test | Statistical significance of shift | p < 0.05 |
| PSI | Magnitude of distribution shift | > 0.25 = severe |

---

## **AWS Infrastructure**

| Service | Configuration | Cost |
|---|---|---|
| S3 | fraud-detection-mlproject-armand (us-east-1) | ~$0.02/GB/month |
| SageMaker Space | ml.t3.large (8GB RAM) | $0.10/hr — stop when idle |
| SageMaker Training | ml.m5.2xlarge (32GB RAM) | $0.46/hr — only during V3 training |
| SageMaker Endpoint | ml.m5.xlarge (16GB RAM) | $0.23/hr — deleted after testing |
| Kinesis Stream | 1 shard, us-east-1 | $0.015/hr — deleted after demo |

**Total project cost: < $10**

---

## **How to Run**

### Prerequisites
```bash
pip install -r requirements.txt
```

Configure AWS credentials:
```bash
aws configure
# Access Key ID     : your-access-key
# Secret Access Key : your-secret-key
# Region            : us-east-1
```

### Step 1 — Provision AWS infrastructure
```bash
bash infrastructure/setup_aws.sh
```

### Step 2 — Upload datasets to S3
```bash
# Download from Kaggle first
kaggle competitions download -c ieee-fraud-detection -p /tmp/
kaggle datasets download -d mlg-ulb/creditcardfraud -p /tmp/

# Upload to S3
aws s3 cp /tmp/train_transaction.csv \
    s3://fraud-detection-mlproject-armand/raw-data/
aws s3 cp /tmp/creditcard.csv \
    s3://fraud-detection-mlproject-armand/raw-data/
```

### Step 3 — Run notebooks in order
```
01_data_ingestion_eda.ipynb   → ml.t3.large
02_model_training.ipynb       → ml.m5.2xlarge (32GB required for V3)
03_deployment.ipynb           → ml.t3.large
04_streaming.ipynb            → ml.t3.large
05_anomaly_detection.ipynb    → ml.t3.large
06_monitoring.ipynb           → ml.t3.large
```

### Step 4 — Cost management
```bash
# Delete Kinesis stream after Notebook 04
aws kinesis delete-stream \
    --stream-name fraud-detection-stream

# Delete SageMaker endpoint after Notebook 03
aws sagemaker delete-endpoint \
    --endpoint-name fraud-xgb-v3

# Stop SageMaker space when not working
# SageMaker Studio → fraud-detection → Stop
```

---

## **Unit Tests**

```bash
# Run all 45 tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

```
tests/test_features.py    14 tests   PASSED
tests/test_model.py       17 tests   PASSED
tests/test_monitoring.py  18 tests   PASSED
─────────────────────────────────────
45 passed in 3.21s
```

Tests cover:
- UID feature creation and NaN handling
- Smoothed target encoding and unseen categories
- Threshold optimisation and model evaluation
- KS test and PSI drift detection
- Alert level logic matching notebook outputs

---

## Contact

**Armand Junior Dongmo Notue**
Arlington, Texas

[![GitHub](https://img.shields.io/badge/GitHub-ARMAND--cod--eng-black)](https://github.com/ARMAND-cod-eng)


[LinkedIn](https://www.linkedin.com/in/notue250/)
---

*Production Fraud Detection Platform on AWS · 2026*
