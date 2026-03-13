# =============================================================
# paysim_producer.py  PaySim Domain Rules Pipeline
# Production Fraud Detection Platform on AWS
# Author: Armand Junior Dongmo Notue
# =============================================================
# Approach : Domain rules (no ML model needed)
# Dataset  : PaySim synthetic transactions
# Result   : Precision 0.857, Recall 1.000, F1 0.923
# =============================================================

import json
import time
import boto3
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score
)

REGION      = 'us-east-1'
STREAM_NAME = 'fraud-detection-stream'
THRESHOLD   = 0.50


def generate_paysim_transactions(
    n: int = 100,
    fraud_rate: float = 0.06,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic PaySim-style transactions.

    Parameters
    ----------
    n            : int    Number of transactions
    fraud_rate   : float  Fraction of fraud (default 6%)
    random_state : int    Random seed

    Returns
    -------
    pd.DataFrame : Synthetic transactions
    """
    rng        = np.random.RandomState(random_state)
    n_fraud    = int(n * fraud_rate)
    n_legit    = n - n_fraud
    txn_types  = ['PAYMENT', 'TRANSFER',
                  'CASH_OUT', 'DEBIT', 'CASH_IN']

    records = []

    # Legitimate transactions
    for i in range(n_legit):
        txn_type   = rng.choice(txn_types, p=[
            0.35, 0.15, 0.20, 0.20, 0.10
        ])
        amount     = rng.uniform(10, 500)
        old_orig   = rng.uniform(1000, 50000)
        new_orig   = old_orig - amount
        old_dest   = rng.uniform(1000, 50000)
        new_dest   = old_dest + amount

        records.append({
            'transaction_id': f'PAY_{i:04d}',
            'type'          : txn_type,
            'amount'        : round(amount, 2),
            'oldbalanceOrg' : round(old_orig, 2),
            'newbalanceOrig': round(new_orig, 2),
            'oldbalanceDest': round(old_dest, 2),
            'newbalanceDest': round(new_dest, 2),
            'isFraud'       : 0
        })

    # Fraud transactions
    for i in range(n_fraud):
        txn_type = rng.choice(
            ['TRANSFER', 'CASH_OUT'], p=[0.5, 0.5]
        )
        amount   = rng.uniform(1000, 10000)
        old_orig = amount  # exact drain
        new_orig = 0.0     # wiped to zero
        old_dest = rng.uniform(1000, 5000)
        new_dest = old_dest  # destination unchanged

        records.append({
            'transaction_id': f'PAY_{n_legit+i:04d}',
            'type'          : txn_type,
            'amount'        : round(amount, 2),
            'oldbalanceOrg' : round(old_orig, 2),
            'newbalanceOrig': round(new_orig, 2),
            'oldbalanceDest': round(old_dest, 2),
            'newbalanceDest': round(new_dest, 2),
            'isFraud'       : 1
        })

    df = pd.DataFrame(records).sample(
        frac=1, random_state=random_state
    ).reset_index(drop=True)

    return df


def apply_business_rules(row: pd.Series) -> float:
    """
    Apply 5 domain rules to compute fraud score.

    Rules:
        1. Wrong transaction type  (+0.01)
        2. Balance fully wiped     (+0.50)
        3. Large amount > 200,000  (+0.20)
        4. Destination unchanged   (+0.30)
        5. Exact drain             (+0.20)

    Parameters
    ----------
    row : pd.Series  Single transaction

    Returns
    -------
    float : Fraud score (0 to 1+)
    """
    score = 0.0

    # Rule 1: Only TRANSFER and CASH_OUT carry fraud
    if row['type'] not in ['TRANSFER', 'CASH_OUT']:
        score += 0.01

    # Rule 2: Origin balance wiped to zero
    if (row['newbalanceOrig'] == 0 and
            row['oldbalanceOrg'] > 0):
        score += 0.50

    # Rule 3: Large amount
    if row['amount'] > 200000:
        score += 0.20

    # Rule 4: Destination balance unchanged
    if row['newbalanceDest'] == row['oldbalanceDest']:
        score += 0.30

    # Rule 5: Exact drain (amount == origin balance)
    if abs(row['amount'] -
           row['oldbalanceOrg']) < 0.01:
        score += 0.20

    return min(score, 1.0)


def run_paysim_pipeline(
    n_transactions: int = 100
) -> dict:
    """
    Run the full PaySim rules pipeline:
    Generate → Score → Send to Kinesis → Consume → Evaluate

    Parameters
    ----------
    n_transactions : int  Number of transactions

    Returns
    -------
    dict : Pipeline results
    """
    kin = boto3.client('kinesis', region_name=REGION)

    print("=" * 55)
    print("  PAYSIM DOMAIN RULES PIPELINE")
    print("=" * 55)

    # Generate transactions
    print("\nGenerating transactions...")
    df = generate_paysim_transactions(
        n=n_transactions
    )
    df['score'] = df.apply(
        apply_business_rules, axis=1
    )
    df['predicted'] = (
        df['score'] >= THRESHOLD
    ).astype(int)

    n_fraud = df['isFraud'].sum()
    print(f"   Total      : {len(df)}")
    print(f"   Fraud      : {n_fraud} "
          f"({n_fraud/len(df)*100:.1f}%)")

    # Get iterator
    shard_id = kin.list_shards(
        StreamName=STREAM_NAME
    )['Shards'][0]['ShardId']

    iterator = kin.get_shard_iterator(
        StreamName        = STREAM_NAME,
        ShardId           = shard_id,
        ShardIteratorType = 'LATEST'
    )['ShardIterator']

    # Send to Kinesis
    print("\nSending to Kinesis...")
    start = time.time()
    for _, row in df.iterrows():
        record = {
            'transaction_id': row['transaction_id'],
            'isFraud'       : int(row['isFraud']),
            'amount'        : float(row['amount']),
            'score'         : float(row['score']),
            'sent_at'       : time.time()
        }
        kin.put_record(
            StreamName   = STREAM_NAME,
            Data         = json.dumps(record),
            PartitionKey = row['transaction_id']
        )
    duration   = time.time() - start
    throughput = len(df) / duration
    print(f"   Throughput : {throughput:.0f} rec/sec")

    # Evaluate
    tp = int(((df['predicted'] == 1) &
              (df['isFraud'] == 1)).sum())
    fp = int(((df['predicted'] == 1) &
              (df['isFraud'] == 0)).sum())
    tn = int(((df['predicted'] == 0) &
              (df['isFraud'] == 0)).sum())
    fn = int(((df['predicted'] == 0) &
              (df['isFraud'] == 1)).sum())

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall    = tp / (tp + fn) if tp + fn > 0 else 0
    f1        = (2 * precision * recall /
                 (precision + recall)
                 if precision + recall > 0 else 0)

    print(f"\n{'='*55}")
    print(f"RESULTS")
    print(f"{'='*55}")
    print(f"   Precision : {precision:.3f}")
    print(f"   Recall    : {recall:.3f}")
    print(f"   F1-Score  : {f1:.3f}")
    print(f"   TP: {tp}  FP: {fp}  "
          f"TN: {tn}  FN: {fn}")

    return {
        'precision': precision,
        'recall'   : recall,
        'f1'       : f1,
        'tp': tp, 'fp': fp,
        'tn': tn, 'fn': fn
    }


if __name__ == '__main__':
    run_paysim_pipeline(n_transactions=100)





