# =============================================================
# model.py — Model Training + Scoring
# Production Fraud Detection Platform on AWS
# Author: Armand Junior Dongmo Notue
# =============================================================

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)


def score_batch(
    batch: pd.DataFrame,
    model: xgb.Booster,
    feature_names: list
) -> np.ndarray:
    """
    Score a batch of transactions using the V3
    XGBoost model.

    Parameters
    ----------
    batch         : pd.DataFrame  Transactions to score
    model         : xgb.Booster   Loaded XGBoost model
    feature_names : list          Feature names in order

    Returns
    -------
    np.ndarray : Fraud probability scores (0 to 1)
    """
    X    = batch[feature_names].fillna(0)\
               .astype(np.float32)
    dmat = xgb.DMatrix(
        pd.DataFrame(X, columns=feature_names)
    )
    return model.predict(dmat)


def find_best_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    start: float = 0.1,
    end: float   = 0.9,
    step: float  = 0.01
) -> tuple:
    """
    Find the decision threshold that maximises F1-Score.

    Parameters
    ----------
    y_true   : np.ndarray  True labels
    y_scores : np.ndarray  Predicted probabilities
    start    : float       Search start (default 0.1)
    end      : float       Search end (default 0.9)
    step     : float       Search step (default 0.01)

    Returns
    -------
    tuple : (best_threshold, best_f1)
    """
    thresholds = np.arange(start, end, step)
    best_f1    = 0.0
    best_thresh = thresholds[0]

    for t in thresholds:
        preds = (y_scores >= t).astype(int)
        f1    = f1_score(
            y_true, preds, zero_division=0
        )
        if f1 > best_f1:
            best_f1     = f1
            best_thresh = t

    return float(best_thresh), float(best_f1)


def evaluate_model(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float = 0.87
) -> dict:
    """
    Compute full evaluation metrics for a set of
    fraud probability scores.

    Parameters
    ----------
    y_true    : np.ndarray  True labels
    y_scores  : np.ndarray  Predicted probabilities
    threshold : float       Decision threshold

    Returns
    -------
    dict : AUC-ROC, AUC-PR, F1, Precision, Recall,
           TP, FP, TN, FN
    """
    y_pred = (y_scores >= threshold).astype(int)
    cm     = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        'auc_roc'  : float(roc_auc_score(
            y_true, y_scores
        )),
        'auc_pr'   : float(average_precision_score(
            y_true, y_scores
        )),
        'f1'       : float(f1_score(
            y_true, y_pred, zero_division=0
        )),
        'precision': float(precision_score(
            y_true, y_pred, zero_division=0
        )),
        'recall'   : float(recall_score(
            y_true, y_pred, zero_division=0
        )),
        'tp'       : int(tp),
        'fp'       : int(fp),
        'tn'       : int(tn),
        'fn'       : int(fn),
        'threshold': threshold
    }


def print_metrics(metrics: dict) -> None:
    """
    Pretty-print evaluation metrics.

    Parameters
    ----------
    metrics : dict  Output of evaluate_model()
    """
    print(f"   AUC-ROC   : {metrics['auc_roc']:.4f}")
    print(f"   AUC-PR    : {metrics['auc_pr']:.4f}")
    print(f"   F1-Score  : {metrics['f1']:.4f}")
    print(f"   Precision : {metrics['precision']:.4f}")
    print(f"   Recall    : {metrics['recall']:.4f}")
    print(f"\n   Confusion Matrix:")
    print(f"      TP : {metrics['tp']:,} "
          f"({metrics['tp']/(metrics['tp']+metrics['fn'])*100:.1f}%"
          f" fraud caught)")
    print(f"      FP : {metrics['fp']:,}")
    print(f"      TN : {metrics['tn']:,}")
    print(f"      FN : {metrics['fn']:,}")



