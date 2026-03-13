# =============================================================
# streaming.py — Kinesis Producer + Consumer
# Production Fraud Detection Platform on AWS
# Author: Armand Junior Dongmo Notue
# =============================================================

import json
import time
import boto3
import numpy as np
import pandas as pd
import xgboost as xgb


def get_kinesis_client(region: str = 'us-east-1'):
    """
    Create and return a Kinesis boto3 client.

    Parameters
    ----------
    region : str  AWS region (default us-east-1)

    Returns
    -------
    boto3 Kinesis client
    """
    return boto3.client('kinesis', region_name=region)


def get_latest_iterator(
    kin,
    stream_name: str,
    iterator_type: str = 'LATEST'
) -> str:
    """
    Get a shard iterator for the first shard
    of a Kinesis stream.

    Parameters
    ----------
    kin           : boto3 Kinesis client
    stream_name   : str  Kinesis stream name
    iterator_type : str  LATEST or TRIM_HORIZON

    Returns
    -------
    str : Shard iterator
    """
    shard_id = kin.list_shards(
        StreamName=stream_name
    )['Shards'][0]['ShardId']

    return kin.get_shard_iterator(
        StreamName        = stream_name,
        ShardId           = shard_id,
        ShardIteratorType = iterator_type
    )['ShardIterator']


def send_transactions(
    kin,
    stream_name: str,
    transactions: pd.DataFrame,
    feature_names: list
) -> dict:
    """
    Send transactions to a Kinesis stream.

    Parameters
    ----------
    kin           : boto3 Kinesis client
    stream_name   : str          Kinesis stream name
    transactions  : pd.DataFrame Transactions to send
    feature_names : list         Feature names to include

    Returns
    -------
    dict : sent_count, fraud_count, throughput
    """
    sent_count  = 0
    fraud_count = 0
    start       = time.time()

    for idx, row in transactions.iterrows():
        features = {
            f: float(row.get(f, 0))
            for f in feature_names
        }
        record = {
            'transaction_id': f'TXN_{idx:06d}',
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
            StreamName   = stream_name,
            Data         = json.dumps(record),
            PartitionKey = f'TXN_{idx:06d}'
        )
        sent_count  += 1
        fraud_count += int(row.get('isFraud', 0))

    duration = time.time() - start

    return {
        'sent_count' : sent_count,
        'fraud_count': fraud_count,
        'duration'   : round(duration, 2),
        'throughput' : round(sent_count / duration, 0)
    }


def consume_and_score(
    kin,
    stream_name: str,
    iterator: str,
    model: xgb.Booster,
    feature_names: list,
    threshold: float = 0.87,
    expected_count: int = 100,
    max_empty: int = 15
) -> list:
    """
    Consume records from Kinesis and score each
    transaction using the fraud detection model.

    Parameters
    ----------
    kin            : boto3 Kinesis client
    stream_name    : str         Stream name
    iterator       : str         Shard iterator
    model          : xgb.Booster Fraud model
    feature_names  : list        Feature names
    threshold      : float       Decision threshold
    expected_count : int         Records to consume
    max_empty      : int         Max empty polls

    Returns
    -------
    list : Scored transaction results
    """
    results     = []
    consumed    = 0
    empty_count = 0

    while (consumed < expected_count
           and empty_count < max_empty):
        response = kin.get_records(
            ShardIterator = iterator,
            Limit         = 10
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

            df_input = pd.DataFrame(
                [data['features']],
                columns=feature_names
            ).astype(np.float32)

            score    = float(
                model.predict(
                    xgb.DMatrix(df_input)
                )[0]
            )
            latency  = (
                recv_time - data['sent_at']
            ) * 1000

            results.append({
                'transaction_id': data['transaction_id'],
                'amount'        : data['amount'],
                'score'         : score,
                'predicted'     : score > threshold,
                'actual'        : bool(data['isFraud']),
                'latency_ms'    : latency
            })
            consumed += 1

    return results


def delete_stream(
    kin,
    stream_name: str
) -> None:
    """
    Delete a Kinesis stream to stop incurring costs.

    Parameters
    ----------
    kin         : boto3 Kinesis client
    stream_name : str  Stream to delete
    """
    try:
        kin.delete_stream(StreamName=stream_name)
        print(f"   Stream deleted: {stream_name}")
        print(f"   Cost savings: $0.015/hour")
    except Exception as e:
        print(f"   Error deleting stream: {e}")



