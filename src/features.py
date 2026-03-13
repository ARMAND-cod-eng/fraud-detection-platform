# =============================================================
# features.py: Feature Engineering
# Production Fraud Detection Platform on AWS
# Author: Armand Junior Dongmo Notue
# =============================================================

import pandas as pd
import numpy as np


def create_uid_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create UID composite key features from card
    and address columns.

    Parameters
    ----------
    df : pd.DataFrame
        Raw transaction dataframe

    Returns
    -------
    pd.DataFrame
        Dataframe with uid1, uid2, uid3 columns added
    """
    for col in [
        'card1', 'card2', 'card3', 'card4',
        'card5', 'card6', 'addr1', 'addr2',
        'P_emaildomain', 'R_emaildomain'
    ]:
        if col in df.columns:
            df[col] = df[col].fillna('unknown').astype(str)

    df['uid1'] = df['card1'] + '_' + df['addr1']
    df['uid2'] = (
        df['card1'] + '_' +
        df['addr1'] + '_' +
        df['P_emaildomain']
    )
    df['uid3'] = df['card1'] + '_' + df['card2']

    return df


def smooth_target_encode(
    train: pd.DataFrame,
    val: pd.DataFrame,
    col: str,
    target: str,
    global_mean: float,
    k: int = 20
) -> tuple:
    """
    Smoothed Bayesian target encoding.

    Prevents overfitting on rare categories by pulling
    rare category means toward the global mean.

    Formula:
        encoded = (count * mean + k * global_mean)
                  / (count + k)

    Parameters
    ----------
    train : pd.DataFrame  Training set
    val   : pd.DataFrame  Validation set
    col   : str           Column to encode
    target: str           Target column name
    global_mean : float   Global target mean
    k     : int           Smoothing factor (default 20)

    Returns
    -------
    tuple : (train_encoded, val_encoded) as np.float32
    """
    stats  = train.groupby(col)[target].agg(
        ['mean', 'count']
    )
    smooth = (
        (stats['count'] * stats['mean'] +
         k * global_mean) /
        (stats['count'] + k)
    )
    train_enc = train[col].map(smooth).fillna(
        global_mean
    ).astype(np.float32)
    val_enc   = val[col].map(smooth).fillna(
        global_mean
    ).astype(np.float32)

    return train_enc, val_enc


def frequency_encode(
    train: pd.DataFrame,
    val: pd.DataFrame,
    cols: list
) -> tuple:
    """
    Frequency encoding — replaces category with
    its relative frequency in the training set.

    Parameters
    ----------
    train : pd.DataFrame  Training set
    val   : pd.DataFrame  Validation set
    cols  : list          Columns to encode

    Returns
    -------
    tuple : (train_df, val_df) with _freq columns added
    """
    for col in cols:
        freq = train[col].value_counts(normalize=True)
        train[f'{col}_freq'] = train[col]\
            .map(freq).fillna(0).astype(np.float32)
        val[f'{col}_freq']   = val[col]\
            .map(freq).fillna(0).astype(np.float32)

    return train, val


def add_target_encoding_features(
    train: pd.DataFrame,
    val: pd.DataFrame,
    target: str = 'isFraud',
    k: int = 20
) -> tuple:
    """
    Apply smoothed target encoding to all UID and
    card/email columns. Also adds count features.

    Parameters
    ----------
    train  : pd.DataFrame  Training set
    val    : pd.DataFrame  Validation set
    target : str           Target column (default isFraud)
    k      : int           Smoothing factor (default 20)

    Returns
    -------
    tuple : (train_df, val_df) with fraud_rate columns added
    """
    global_mean = train[target].mean()

    target_cols = [
        'uid1', 'uid2', 'uid3',
        'card1', 'card2', 'addr1',
        'P_emaildomain', 'R_emaildomain'
    ]

    for col in target_cols:
        if col not in train.columns:
            continue

        t_enc, v_enc = smooth_target_encode(
            train, val, col, target, global_mean, k
        )
        train[f'{col}_fraud_rate'] = t_enc
        val[f'{col}_fraud_rate']   = v_enc

        cnt = train.groupby(col)[target].count()
        train[f'{col}_count'] = train[col]\
            .map(cnt).fillna(0).astype(np.int32)
        val[f'{col}_count'] = val[col]\
            .map(cnt).fillna(0).astype(np.int32)

    return train, val


def get_feature_names(df: pd.DataFrame) -> list:
    """
    Returns the final list of numeric feature columns,
    excluding target and string columns.

    Parameters
    ----------
    df : pd.DataFrame  Feature dataframe

    Returns
    -------
    list : Feature column names
    """
    drop_cols = [
        'TransactionID', 'isFraud',
        'uid1', 'uid2', 'uid3'
    ]
    obj_cols  = df.select_dtypes(
        include=['object']
    ).columns.tolist()
    drop_cols = list(set(drop_cols + obj_cols))

    return [
        c for c in df.columns
        if c not in drop_cols
    ]



