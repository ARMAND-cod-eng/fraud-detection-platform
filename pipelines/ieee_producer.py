# =============================================================
# ieee_producer.py : IEEE-CIS V3 XGBoost Pipeline
# Production Fraud Detection Platform on AWS
# Author: Armand Junior Dongmo Notue
# =============================================================
# Approach : V3 XGBoost (AUC 0.9622, 626 features)
# Dataset  : IEEE-CIS real-world transactions
# Result   : Precision 1.000, Recall 0.850, F1 0.919
# =============================================================

import json
import time
import boto3
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    f1_score, precision_score, recall_score
)

REGION      = 'us-east-1'
STREAM_NAME = 'fraud-detection-stream'
BUCKET      = 'fraud-detection-mlproject-armand'
THRESHOLD   = 0.87


def load_model() -> tuple:
    """
    Load V3 XGBoost model and feature names from S3.

    Returns
    -------
    tuple : (xgb.Booster, list of feature names)
    """
    import os
    s3 = boto3.client('s3', region_name=REGION)

    os.makedirs('/tmp/v3', exist_ok=True)

    for key, name in [
        ('models/v3/xgb_model_v3fix.json',
         'model.json'),
        ('models/v3/feature_names_v3fix.json',
         'features.json'),
    ]:
        s3.download_file(
            BUCKET, key, f'/tmp/v3/{name}'
        )

    model = xgb.Booster()
    model.load_model('/tmp/v3/model.json')

    with open('/tmp/v3/features.json') as f:
        feature_names = json.load(f)

    print(f"   Model loaded!")
    print(f"   Features  : {len(feature_names)}")
    print(f"   Threshold : {THRESHOLD}")

    return model, feature_names


def load_sample_transactions(
    feature_names: list,
    n_fraud: int = 20,
    n_legit: int = 80
) -> pd.DataFrame:
    """
    Load a balanced sample from df_features_v3.csv.

    Parameters
    ----------
    feature_names : list  V3 feature names
    n_fraud       : int   Number of fraud rows
    n_legit       : int   Number of legit rows

    Returns
    -------
    pd.DataFrame : Balanced sample
    """
    import os
    s3 = boto3.client('s3', region_name=REGION)

    local_path = '/tmp/df_features_v3.csv'
    if not os.path.exists(local_path):
        print("   Downloading df_features_v3.csv...")
        s3.download_file(
            BUCKET,
            'processed-data/df_features_v3.csv',
            local_path
        )

    df        = pd.read_csv(local_path, nrows=50000)
    fraud_df  = df[df['isFraud'] == 1].head(n_fraud)
    normal_df = df[df['isFraud'] == 0].head(n_legit)

    stream_df = pd.concat(
        [fraud_df, normal_df]
    ).sample(frac=1, random_state=42)\
     .reset_index(drop=True)

    # Fill missing features
    for feat in feature_names:
        if feat not in stream_df.columns:
            stream_df[feat] = 0.0

    print(f"   Loaded: {len(stream_df)} transactions")
    print(f"   Fraud : {n_fraud} | Legit: {n_legit}")

    return stream_df


def score_transaction(
    features: dict,
    model: xgb.Booster,
    feature_names: list
) -> float:
    """
    Score a single transaction.

    Parameters
    ----------
    features      : dict         Feature values
    model         : xgb.Booster  Loaded model
    feature_names : list         Feature names

    Returns
    -------
    float : Fraud probability (0 to 1)
    """
    df_input = pd.DataFrame(
        [[float(features.get(f, 0))
          for f in feature_names]],
        columns=feature_names
    ).astype(np.float32)

    return float(
        model.predict(xgb.DMatrix(df_input))[0]
    )


def run_ieee_pipeline() -> dict:
    """
    Run the full IEEE-CIS V3 XGBoost pipeline:
    Load model → Load data → Get iterator →
    Send → Consume + Score → Evaluate

    Returns
    -------
    dict : Pipeline results
    """
    kin = boto3.client('kinesis', region_name=REGION)

    print("=" * 55)
    print("  IEEE-CIS V3 XGBOOST PIPELINE")
    print("=" * 55)

    # Load model
    print("\nStep 1: Loading V3 model...")
    model, feature_names = load_model()

    # Load transactions
    print("\nStep 2: Loading transactions...")
    stream_df = load_sample_transactions(
        feature_names
    )

    # Get LATEST iterator
    print("\nStep 3: Getting Kinesis iterator...")
    shard_id = kin.list_shards(
        StreamName=STREAM_NAME
    )['Shards'][0]['ShardId']

    iterator = kin.get_shard_iterator(
        StreamName        = STREAM_NAME,
        ShardId           = shard_id,
        ShardIteratorType = 'LATEST'
    )['ShardIterator']
    print(f"   Iterator ready!")

    # Send transactions
    print("\nStep 4: Sending transactions...")
    start      = time.time()
    sent_count = 0

    for idx, row in stream_df.iterrows():
        features = {
            f: float(row.get(f, 0))
            for f in feature_names
        }
        record = {
            'transaction_id': f'IEEE_{idx:06d}',
            'isFraud'       : int(
                row.get('isFraud', 0)
            ),
            'amount'        : float(
                row.get('TransactionAmt', 0)
            ),
            'features'      : features,
            'sent_at'       : time.time()
        }
        kin.put_record(
            StreamName   = STREAM_NAME,
            Data         = json.dumps(record),
            PartitionKey = f'IEEE_{idx:06d}'
        )
        sent_count += 1

    throughput = sent_count / (time.time() - start)
    print(f"   Sent       : {sent_count} transactions")
    print(f"   Throughput : {throughput:.0f} rec/sec")

    # Consume + Score
    print("\nStep 5: Consuming and scoring...")
    print(f"\n   {'Transaction':<12} {'Amount':>10} "
          f"{'Score':>6} {'Decision':<10} "
          f"{'Actual':<8} {'Match'}")
    print(f"   {'-'*58}")

    results     = []
    consumed    = 0
    empty_count = 0

    while consumed < 100 and empty_count < 15:
        response = kin.get_records(
            ShardIterator=iterator, Limit=10
        )
        iterator = response['NextShardIterator']
        records  = response['Records']

        if len(records) == 0:
            empty_count += 1
            time.sleep(0.3)
            continue

        empty_count = 0

        for record in records:
            recv_time = time.time()
            data      = json.loads(
                record['Data'].decode('utf-8')
            )
            if 'features' not in data:
                continue

            score    = score_transaction(
                data['features'], model, feature_names
            )
            latency  = (
                recv_time - data['sent_at']
            ) * 1000
            is_fraud = score > THRESHOLD
            actual   = bool(data['isFraud'])
            match    = "OK" if is_fraud == actual \
                       else "MISS"
            consumed += 1

            print(f"   {data['transaction_id']:<12} "
                  f"${data['amount']:>9.2f} "
                  f"{score:>6.3f} "
                  f"{'FRAUD' if is_fraud else 'LEGIT':<10} "
                  f"{'FRAUD' if actual else 'LEGIT':<8} "
                  f"{match}")

            results.append({
                'predicted': is_fraud,
                'actual'   : actual,
                'score'    : score,
                'latency'  : latency
            })

    # Evaluate
    df_r = pd.DataFrame(results)
    tp   = int(((df_r['predicted']) &
                (df_r['actual'])).sum())
    fp   = int(((df_r['predicted']) &
                (~df_r['actual'])).sum())
    tn   = int(((~df_r['predicted']) &
                (~df_r['actual'])).sum())
    fn   = int(((~df_r['predicted']) &
                (df_r['actual'])).sum())

    precision = tp / (tp+fp) if tp+fp > 0 else 0
    recall    = tp / (tp+fn) if tp+fn > 0 else 0
    f1        = (2 * precision * recall /
                 (precision + recall)
                 if precision + recall > 0 else 0)
    latencies = [r['latency'] for r in results]

    print(f"\n{'='*55}")
    print(f"RESULTS")
    print(f"{'='*55}")
    print(f"   Precision : {precision:.3f}")
    print(f"   Recall    : {recall:.3f}")
    print(f"   F1-Score  : {f1:.3f}")
    print(f"   TP: {tp}  FP: {fp}  "
          f"TN: {tn}  FN: {fn}")
    print(f"   Avg Latency: "
          f"{np.mean(latencies):.0f}ms")

    return {
        'precision': precision,
        'recall'   : recall,
        'f1'       : f1,
        'tp': tp, 'fp': fp,
        'tn': tn, 'fn': fn
    }


if __name__ == '__main__':
    run_ieee_pipeline()



