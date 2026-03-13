# infrastructure/ — AWS Setup Scripts

This folder contains scripts to provision and manage
the AWS infrastructure used by the Fraud Detection Platform.

---

## Files

| File | Purpose |
|---|---|
| `setup_aws.sh` | Create S3 bucket, IAM user, Kinesis permissions |
| `setup_sagemaker.py` | SageMaker utilities + infrastructure status check |

---

## AWS Architecture
```
┌─────────────────────────────────────────────────┐
│                  AWS Account                     │
│                                                  │
│  ┌──────────┐    ┌────────────┐    ┌──────────┐ │
│  │    S3    │    │ SageMaker  │    │ Kinesis  │ │
│  │          │    │            │    │          │ │
│  │ raw-data │    │ JupyterLab │    │  stream  │ │
│  │ features │    │ Training   │    │ 1 shard  │ │
│  │ models   │    │ Endpoint   │    │          │ │
│  └──────────┘    └────────────┘    └──────────┘ │
│                                                  │
│  ┌──────────────────────────────────────────┐   │
│  │                   IAM                    │   │
│  │  ml-engineer user (AdministratorAccess)  │   │
│  │  SageMaker execution role + Kinesis      │   │
│  └──────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

---

## Quick Start

### Step 1 — Prerequisites
```bash
# Install AWS CLI
pip install awscli

# Configure with ml-engineer credentials
aws configure
# AWS Access Key ID     : your-access-key
# AWS Secret Access Key : your-secret-key
# Default region        : us-east-1
# Default output format : json
```

### Step 2 — Create S3 + IAM
```bash
bash setup_aws.sh
```

Expected output:
```
=============================================
  AWS Infrastructure Setup
  Fraud Detection Platform
=============================================

Step 1: Creating S3 bucket...
   Bucket created: s3://fraud-detection-mlproject-armand
   Folders: raw-data/, processed-data/, models/

Step 2: Creating IAM user...
   IAM user created: ml-engineer

Step 3: Creating access keys...
   Run: aws iam create-access-key --user-name ml-engineer

Step 4: Kinesis permissions for SageMaker...
   Kinesis permissions added!

=============================================
  Setup Complete!
=============================================
```

### Step 3 — Check infrastructure status
```bash
python setup_sagemaker.py
```

Expected output:
```
=======================================================
  INFRASTRUCTURE STATUS CHECK
=======================================================

   S3 Bucket     : OK
   Bucket name   : fraud-detection-mlproject-armand
   Folders       : ['models/', 'processed-data/', 'raw-data/']

   SageMaker     : OK
   Domain        : QuickSetupDomain-20260304T184323
   Status        : InService

   Kinesis       : No active streams
=======================================================
```

### Step 4 — Create Kinesis stream (when needed)
```python
from setup_sagemaker import create_kinesis_stream

stream = create_kinesis_stream(
    stream_name='fraud-detection-stream',
    shard_count=1
)
# Cost: $0.015/hour — delete after use!
```

### Step 5 — Delete Kinesis stream (after use)
```python
from setup_sagemaker import delete_kinesis_stream

delete_kinesis_stream('fraud-detection-stream')
# Saving $0.015/hour!
```

---

## S3 Data Lake Structure
```
s3://fraud-detection-mlproject-armand/
│
├── raw-data/
│   ├── train_transaction.csv     (683 MB — IEEE-CIS)
│   ├── train_identity.csv        (26 MB  — IEEE-CIS)
│   └── creditcard.csv            (144 MB — ULB)
│
├── processed-data/
│   ├── df_features_v2.csv        (621 features)
│   └── df_features_v3.csv        (626 features — PRODUCTION)
│
└── models/
    ├── v3/
    │   ├── xgb_model_v3fix.json       (XGBoost V3)
    │   ├── lgb_model_v3fix.txt        (LightGBM V3)
    │   └── feature_names_v3fix.json   (626 feature names)
    └── v4/
        ├── xgb_v4.json
        ├── lgb_v4.txt
        ├── cat_v4.cbm
        └── meta_v4.pkl
```

---

## Cost Management

| Service | Cost | Action |
|---|---|---|
| S3 storage | ~$0.02/GB/month | Keep — minimal cost |
| SageMaker ml.t3.large | $0.10/hour | **Stop when idle** |
| SageMaker ml.m5.xlarge | $0.23/hour | **Stop when idle** |
| Kinesis stream | $0.015/hour | **Delete after each demo** |
| SageMaker endpoint | $0.28/hour | **Delete after testing** |

**Total project cost: < $10**

### Cost saving commands
```bash
# Check for running endpoints (costs money!)
aws sagemaker list-endpoints \
    --status-filter InService \
    --region us-east-1

# Delete an endpoint
aws sagemaker delete-endpoint \
    --endpoint-name fraud-xgb-v3 \
    --region us-east-1

# Check Kinesis streams (costs money!)
aws kinesis list-streams --region us-east-1

# Delete Kinesis stream
aws kinesis delete-stream \
    --stream-name fraud-detection-stream \
    --region us-east-1
```

---

## Region
All resources deployed in **us-east-1** (N. Virginia).

---

*Production Fraud Detection Platform on AWS · 2026*
*Armand Junior Dongmo Notue · Arlington, Texas*






