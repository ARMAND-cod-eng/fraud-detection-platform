# =============================================================
# test_features.py : Unit Tests for Feature Engineering
# Production Fraud Detection Platform on AWS
# Author: Armand Junior Dongmo Notue
# =============================================================
# Run: pytest tests/test_features.py -v
# =============================================================

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
)))

from src.features import (
    create_uid_features,
    smooth_target_encode,
    frequency_encode,
    get_feature_names
)


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Sample transaction dataframe for testing."""
    return pd.DataFrame({
        'TransactionID'  : [1, 2, 3, 4, 5],
        'isFraud'        : [0, 1, 0, 0, 1],
        'TransactionAmt' : [100, 200, 50, 300, 75],
        'card1'          : ['A', 'B', 'A', 'C', 'B'],
        'card2'          : ['X', 'Y', 'X', 'Z', 'Y'],
        'addr1'          : ['100', '200', '100',
                            '300', '200'],
        'P_emaildomain'  : ['gmail.com', 'yahoo.com',
                            'gmail.com', 'hotmail.com',
                            'yahoo.com'],
        'R_emaildomain'  : ['gmail.com', 'yahoo.com',
                            'gmail.com', 'hotmail.com',
                            'yahoo.com'],
    })


@pytest.fixture
def train_val_df():
    """Train/val split for encoding tests."""
    np.random.seed(42)
    n = 100
    train = pd.DataFrame({
        'card1'   : np.random.choice(
            ['A', 'B', 'C', 'D'], n
        ),
        'isFraud' : np.random.randint(0, 2, n)
    })
    val = pd.DataFrame({
        'card1'   : np.random.choice(
            ['A', 'B', 'C', 'E'], 20  # E is unseen
        ),
        'isFraud' : np.random.randint(0, 2, 20)
    })
    return train, val


# ── Tests: create_uid_features ────────────────────────────────

class TestCreateUidFeatures:

    def test_uid_columns_created(self, sample_df):
        """uid1, uid2, uid3 must be created."""
        result = create_uid_features(sample_df.copy())
        assert 'uid1' in result.columns
        assert 'uid2' in result.columns
        assert 'uid3' in result.columns

    def test_uid1_format(self, sample_df):
        """uid1 = card1_addr1."""
        result = create_uid_features(sample_df.copy())
        assert result['uid1'].iloc[0] == 'A_100'
        assert result['uid1'].iloc[1] == 'B_200'

    def test_uid2_format(self, sample_df):
        """uid2 = card1_addr1_P_emaildomain."""
        result = create_uid_features(sample_df.copy())
        assert result['uid2'].iloc[0] == \
            'A_100_gmail.com'

    def test_uid3_format(self, sample_df):
        """uid3 = card1_card2."""
        result = create_uid_features(sample_df.copy())
        assert result['uid3'].iloc[0] == 'A_X'

    def test_nan_handling(self):
        """NaN values must be replaced with 'unknown'."""
        df = pd.DataFrame({
            'card1'         : ['A', None],
            'card2'         : ['X', 'Y'],
            'addr1'         : ['100', None],
            'P_emaildomain' : ['gmail.com', None],
            'R_emaildomain' : ['gmail.com', None],
        })
        result = create_uid_features(df)
        assert 'unknown' in result['uid1'].iloc[1]

    def test_original_columns_preserved(self, sample_df):
        """Original columns must not be removed."""
        result = create_uid_features(sample_df.copy())
        assert 'card1' in result.columns
        assert 'addr1' in result.columns


# ── Tests: smooth_target_encode ───────────────────────────────

class TestSmoothTargetEncode:

    def test_output_shape(self, train_val_df):
        """Output length must match input length."""
        train, val     = train_val_df
        global_mean    = train['isFraud'].mean()
        train_enc, val_enc = smooth_target_encode(
            train, val, 'card1', 'isFraud',
            global_mean, k=20
        )
        assert len(train_enc) == len(train)
        assert len(val_enc)   == len(val)

    def test_output_dtype(self, train_val_df):
        """Output must be float32."""
        train, val  = train_val_df
        global_mean = train['isFraud'].mean()
        train_enc, val_enc = smooth_target_encode(
            train, val, 'card1', 'isFraud',
            global_mean, k=20
        )
        assert train_enc.dtype == np.float32
        assert val_enc.dtype   == np.float32

    def test_unseen_category_gets_global_mean(
        self, train_val_df
    ):
        """Unseen categories must map to global mean."""
        train, val  = train_val_df
        global_mean = train['isFraud'].mean()
        _, val_enc  = smooth_target_encode(
            train, val, 'card1', 'isFraud',
            global_mean, k=20
        )
        # Category 'E' is unseen in train
        # Its encoded value must equal global_mean
        e_mask = val['card1'] == 'E'
        if e_mask.sum() > 0:
            e_values = val_enc[e_mask].values
            assert np.allclose(
                e_values,
                global_mean,
                atol=1e-5
            )

    def test_smoothing_pulls_toward_global_mean(self):
        """Rare categories must be pulled toward mean."""
        train = pd.DataFrame({
            'card1'  : ['A'] * 200 + ['B'] * 2,
            'isFraud': [1] * 200 + [0, 0]
        })
        val         = pd.DataFrame({'card1': ['B']})
        global_mean = train['isFraud'].mean()
        _, val_enc  = smooth_target_encode(
            train, val, 'card1', 'isFraud',
            global_mean, k=20
        )
        # B has 0 fraud rate but only 2 samples
        # Smoothed value must be above 0
        assert val_enc.iloc[0] > 0

    def test_no_missing_values(self, train_val_df):
        """Output must have no NaN values."""
        train, val  = train_val_df
        global_mean = train['isFraud'].mean()
        train_enc, val_enc = smooth_target_encode(
            train, val, 'card1', 'isFraud',
            global_mean, k=20
        )
        assert not train_enc.isna().any()
        assert not val_enc.isna().any()


# ── Tests: frequency_encode ───────────────────────────────────

class TestFrequencyEncode:

    def test_freq_columns_created(self, train_val_df):
        """_freq columns must be created."""
        train, val = train_val_df
        train, val = frequency_encode(
            train.copy(), val.copy(), ['card1']
        )
        assert 'card1_freq' in train.columns
        assert 'card1_freq' in val.columns

    def test_freq_values_sum_to_one(self, train_val_df):
        """Unique freq values must sum to 1.0."""
        train, val = train_val_df
        train, val = frequency_encode(
            train.copy(), val.copy(), ['card1']
        )
        total = train.groupby('card1')[
            'card1_freq'
        ].first().sum()
        assert abs(total - 1.0) < 1e-5

    def test_freq_dtype(self, train_val_df):
        """Frequency values must be float32."""
        train, val = train_val_df
        train, val = frequency_encode(
            train.copy(), val.copy(), ['card1']
        )
        assert train['card1_freq'].dtype == np.float32

    def test_unseen_category_gets_zero(
        self, train_val_df
    ):
        """Unseen categories in val get frequency 0."""
        train, val = train_val_df
        train, val = frequency_encode(
            train.copy(), val.copy(), ['card1']
        )
        e_mask = val['card1'] == 'E'
        if e_mask.sum() > 0:
            assert (
                val.loc[e_mask, 'card1_freq'] == 0
            ).all()


# ── Tests: get_feature_names ──────────────────────────────────

class TestGetFeatureNames:

    def test_excludes_target(self, sample_df):
        """isFraud must not be in feature names."""
        features = get_feature_names(sample_df)
        assert 'isFraud' not in features

    def test_excludes_transaction_id(self, sample_df):
        """TransactionID must not be in feature names."""
        features = get_feature_names(sample_df)
        assert 'TransactionID' not in features

    def test_excludes_object_columns(self, sample_df):
        """String columns must not be in feature names."""
        features = get_feature_names(sample_df)
        obj_cols = sample_df.select_dtypes(
            include=['object']
        ).columns.tolist()
        for col in obj_cols:
            assert col not in features

    def test_numeric_columns_included(self, sample_df):
        """Numeric columns must be in feature names."""
        features = get_feature_names(sample_df)
        assert 'TransactionAmt' in features

    def test_returns_list(self, sample_df):
        """Must return a list."""
        features = get_feature_names(sample_df)
        assert isinstance(features, list)






