# dashboard/  Monitoring Charts + Visualisations

This folder contains the visual outputs from all
6 notebooks of the Production Fraud Detection Platform.

> Images will be added progressively.
> All charts are reproducible by running the notebooks.

---

## Chart Index

### Notebook 01 — EDA
Generated in: `notebooks/01_data_ingestion_eda.ipynb`

| Chart | Description |
|---|---|
| Class distribution | 3.5% fraud rate, 28:1 imbalance |
| Amount distribution | Fraud vs legitimate transaction amounts |
| Feature correlation | Top features correlated with fraud |
| Time distribution | Fraud patterns over 48-hour window |

---

### Notebook 02 — Model Training
Generated in: `notebooks/02_model_training.ipynb`

| Chart | Description |
|---|---|
| V1→V4 progression | AUC-ROC and F1 across all versions |
| Learning curves | XGBoost + LightGBM training curves |
| Feature importance | Top 20 features by gain |
| Precision-Recall curve | AUC-PR 0.7984 for V3 ensemble |
| Confusion matrix | TP=2889, FP=636 at threshold 0.87 |

**Key result:**
```
Version   AUC-ROC   F1      Features
V1        0.9533    0.4288  142
V2        0.9589    0.4817  621
V3        0.9622    0.7545  626  <- PRODUCTION
V4        0.9583    0.7306  626
```

---

### Notebook 03 — SageMaker Deployment
Generated in: `notebooks/03_deployment.ipynb`

| Chart | Description |
|---|---|
| Latency distribution | Avg 36.9ms, P99 131.7ms |
| Score distribution | V1 vs V3 fraud score comparison |
| Test results | 3/3 fraud caught in endpoint testing |

**Endpoint details:**
```
Instance  : ml.m5.xlarge
Latency   : avg 36.9ms
Container : xgboost:1.7-1 (AWS built-in)
Status    : Deleted after testing (zero cost)
```

---

### Notebook 04 — Real-Time Streaming
Generated in: `notebooks/04_streaming.ipynb`

| Chart | Description |
|---|---|
| PaySim score distribution | Rule-based fraud scores |
| IEEE-CIS score distribution | ML model fraud scores |
| Streaming comparison | Rules vs ML side by side |

**Results:**
```
Approach         Precision  Recall  F1
PaySim Rules     0.857      1.000   0.923
IEEE-CIS V3      1.000      0.850   0.919
```

---

### Notebook 05 — Anomaly Detection
Generated in: `notebooks/05_anomaly_detection.ipynb`

| Chart | Description |
|---|---|
| ULB EDA | 577:1 imbalance, PCA feature separation |
| Isolation Forest | Score distribution + PR curve |
| Autoencoder | Training loss + reconstruction error |
| One-Class SVM | Anomaly score distribution |
| Final comparison | All 3 methods side by side |

**Results:**
```
Method             AUC-ROC   F1      Fraud Caught
Isolation Forest   0.9473    0.2169  109/492 (22%)
Autoencoder        0.9420    0.4140  267/492 (54%)
One-Class SVM      0.9490    0.3806  278/492 (57%)
```

**Key insight:**
Autoencoder achieves 45.9x reconstruction error
separation between fraud and legitimate transactions —
the strongest unsupervised signal in this project.

---

### Notebook 06 — Model Monitoring
Generated in: `notebooks/06_monitoring.ipynb`

| Chart | Description |
|---|---|
| Baseline distributions | Month 1 feature distributions |
| Drift simulation | Score distributions across 6 months |
| PSI heatmap | Feature drift intensity per month |
| Monitoring dashboard | Full 6-month overview |

**Alert timeline:**
```
Month 1  GREEN   Baseline — all metrics normal
Month 2  ORANGE  PSI=0.11 moderate drift
Month 3  ORANGE  PSI=0.19 moderate drift
Month 4  ORANGE  PSI=0.16 moderate drift
Month 5  RED     PSI=0.33 SEVERE drift → Retrain!
Month 6  RED     F1 dropped 40% from baseline
```


---

*Production Fraud Detection Platform on AWS · 2026*
*Armand Junior Dongmo Notue · Arlington, Texas*






