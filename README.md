# **Production Fraud Detection Platform on AWS**

> End-to-end ML platform covering data engineering, model training, real-time deployment,
> streaming pipelines, unsupervised anomaly detection, and production monitoring 
> built entirely on AWS.

**Author:** Armand Junior Dongmo Notue · Arlington, Texas  
**Stack:** Python · XGBoost · LightGBM · CatBoost · AWS SageMaker · Amazon Kinesis · S3 · IAM

---

## **Project Overview**

This project builds a production-grade fraud detection platform across 6 notebooks,
3 datasets, and the full ML lifecycle, from raw CSV to deployed endpoint to drift alerting.

Most ML portfolios stop at model training. This project covers everything that happens after:
deployment, real-time inference, unsupervised detection, and monitoring.

---

## **Results Summary**

| Notebook | Topic | Key Result |
|---|---|---|
| 01 | Data Ingestion + EDA | 590,540 transactions, 434 features engineered |
| 02 | Model Training V1→V4 | V3 XGBoost **AUC 0.9622**, F1 0.7545 |
| 03 | SageMaker Deployment | Real-time endpoint, avg **36.9ms latency** |
| 04 | Real-Time Streaming | Kinesis pipeline **F1 0.919**, zero false alarms |
| 05 | Anomaly Detection | Autoencoder **AUC 0.942**, 45.9x separation ratio |
| 06 | Model Monitoring | Drift detected Month 5, **PSI 0.33 alert triggered** |

---

## **Datasets**

| Dataset | Source | Transactions | Fraud Rate | Use |
|---|---|---|---|---|
| **IEEE-CIS Fraud Detection** | Kaggle | 590,540 | 3.5% | Supervised training |
| **PaySim Synthetic** | Kaggle | 6,362 | 7.5% | Streaming rules demo |
| **ULB Credit Card Fraud** | Kaggle | 284,807 | 0.17% | Unsupervised detection |

---

## **Architecture**

```
Raw Data (S3)
     │
     ▼
Feature Engineering (Notebook 01)
  - 434 base features
  - UID composite keys
  - Smoothed target encoding
  - Frequency encoding
  - 626 final features
     │
     ▼
Model Training (Notebook 02)
  V1 → V2 → V3 (PRODUCTION) → V4
  XGBoost + LightGBM + CatBoost Ensemble
     │
     ▼
SageMaker Deployment (Notebook 03)
  Real-time endpoint (ml.m5.xlarge)
  36.9ms average latency
     │
     ▼
Real-Time Streaming (Notebook 04)
  Amazon Kinesis stream
  PaySim domain rules (F1 0.923)
  IEEE-CIS V3 XGBoost (F1 0.919)
     │
     ▼
Anomaly Detection (Notebook 05)
  ULB dataset zero labels used
  Isolation Forest | Autoencoder | One-Class SVM
     │
     ▼
Model Monitoring (Notebook 06)
  6-month drift simulation
  KS Test + PSI feature drift
  GREEN / ORANGE / RED alerting
```

---

## **Model Progression**

| Version | AUC-ROC | AUC-PR | F1 | Features | Notes |
|---|---|---|---|---|---|
| V1 | 0.9533 | 0.6993 | 0.4288 | 142 | Baseline |
| V2 | 0.9589 | 0.7361 | 0.4817 | 621 | + V-features |
| **V3** | **0.9622** | **0.7984** | **0.7545** | **626** | **+ Target encoding → PRODUCTION** |
| V4 | 0.9583 | 0.7737 | 0.7306 | 626 | + CatBoost ensemble |

V3 is the production model. The addition of smoothed Bayesian target encoding
(`uid_fraud_rate`, `card1_fraud_rate`, `P_emaildomain_fraud_rate`) drove F1
from 0.48 to 0.75 a 57% improvement.

---

## **Notebook Details**

### **Notebook 01: Data Ingestion + EDA**
- Loads IEEE-CIS train_transaction.csv (683MB) and train_identity.csv (26MB) from S3
- Merges on TransactionID, handles missing values
- Engineers 434 features: card aggregates, email domains, time features
- EDA: class imbalance (3.5%), amount distributions, correlation analysis

### **Notebook 02: Model Training**
- V1: XGBoost + LightGBM on 142 base features
- V2: Adds 479 V-features from IEEE-CIS identity data
- V3: Adds UID composite keys + smoothed Bayesian target encoding (k=20)
- V4: Adds CatBoost to create a 3-model stacking ensemble
- Optimal threshold search: 0.87 maximises F1 on validation set

### **Notebook 03: SageMaker Deployment**
- Packages V3 XGBoost as built-in container (xgboost:1.7-1)
- Deploys to ml.m5.xlarge real-time endpoint
- Latency test: avg 36.9ms, P99 131.7ms
- Endpoint deleted after testing (zero ongoing cost)

### **Notebook 04: Real-Time Streaming**
- Creates Amazon Kinesis stream (1 shard, fraud-detection-stream)
- Approach A: PaySim + 5 domain rules → Precision 0.857, Recall 1.000, F1 0.923
- Approach B: IEEE-CIS + V3 XGBoost → Precision 1.000, Recall 0.850, F1 0.919
- Key finding: zero false positives in ML approach — model never wrongly blocks a card
- Stream deleted after testing (zero ongoing cost)

### **Notebook 05: Anomaly Detection**
- Dataset: ULB (284,807 transactions, 0.17% fraud, 577:1 imbalance)
- Isolation Forest: AUC 0.9473, F1 0.2169, trained in 3.8s
- Autoencoder (NumPy, 30→16→8→16→30): AUC 0.9420, F1 0.4140, 45.9x separation ratio
- One-Class SVM: AUC 0.9490, F1 0.3806, highest recall (56.5%)
- All three trained on legitimate transactions only — no fraud labels used

### **Notebook 06: Model Monitoring**
- Simulates 6 months of production traffic using monthly batches
- Injects 4 drift scenarios: small-amount fraud, card network compromise, email domain shift, full concept drift
- Feature drift: KS Test (p < 0.05) + PSI (> 0.25 = severe)
- Month 5 triggers RED alert: P_emaildomain_fraud_rate PSI = 0.33
- Month 6 triggers RED alert: F1 drops 40% from baseline
- Dashboard: AUC, F1, Recall, PSI tracked monthly with colour-coded alerts

---

## **AWS Infrastructure**

| Service | Usage | Cost Management |
|---|---|---|
| **S3** | Data lake — raw data, processed features, models | Always-on, minimal cost |
| **SageMaker** | JupyterLab space + model training + endpoint | Space stopped when idle |
| **Kinesis** | Real-time transaction stream | Deleted after each demo |
| **IAM** | ml-engineer user with least-privilege access | No cost |

**Total AWS spend for this project: < $10**

---

## **Key Technical Decisions**

**Why V3 over V4?**
V4 adds CatBoost to create a 3-model ensemble. AUC improves marginally (0.9627→0.9627)
but F1 drops from 0.7545 to 0.7306. Added complexity not justified by the result.

**Why smoothed target encoding?**
Standard target encoding leaks information for rare categories — a card seen once in
training gets an unreliable fraud rate. Smoothed Bayesian encoding (k=20) pulls rare
categories toward the global mean, preventing overfitting on rare cards/emails.

**Why threshold 0.87?**
In fraud detection, false positives (blocking legitimate customers) are more damaging
to business than false negatives (missing some fraud). The 0.87 threshold maximises F1
while maintaining Precision > 0.80.

**Why Autoencoder over Isolation Forest for anomaly detection?**
The Autoencoder's 45.9x reconstruction error separation (fraud vs legitimate) confirms
it learned the geometry of normal transactions most precisely. Isolation Forest is faster
(3.8s) but has 3x lower AUC-PR. For production with GPU, Autoencoder wins.

---

## **Repository Structure**

```
fraud-detection-platform/
│
├── 01_data_ingestion_eda.ipynb
├── 02_model_training.ipynb
├── 03_deployment.ipynb
├── 04_streaming.ipynb
├── 05_anomaly_detection.ipynb
├── 06_monitoring.ipynb
│
├── README.md
└── requirements.txt
```

---

## **Requirements**

```
pandas>=2.0.0
numpy>=1.26.0
xgboost>=2.0.0
lightgbm>=4.0.0
catboost>=1.2.0
scikit-learn>=1.3.0
boto3>=1.34.0
matplotlib>=3.7.0
seaborn>=0.13.0
scipy>=1.11.0
```

---

## **How to Run**

1. **AWS Setup** : Create S3 bucket, SageMaker domain, IAM user
2. **Upload data** : Load 3 datasets to `s3://your-bucket/raw-data/`
3. **Run notebooks** in order: 01 → 02 → 03 → 04 → 05 → 06
4. **Instance recommendations:**
   - Notebooks 01, 04, 05, 06: `ml.t3.large` (cost-efficient)
   - Notebook 02 (V3 training): `ml.m5.2xlarge` (32GB RAM required)
   - Notebook 03 (deployment): `ml.t3.large` + deploys to `ml.m5.xlarge`

---

## Contact

**Armand Junior Dongmo Notue**  
Arlington, Texas  
[LinkedIn](https://www.linkedin.com/in/notue250/) · [GitHub](https://github.com/ARMAND-cod-eng/)

---

*Production Fraud Detection Platform on AWS · 2026*
