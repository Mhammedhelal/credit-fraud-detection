import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from data_utils.time_features import add_time_features
from data_utils.amount_features import add_amount_features
from data_utils.v_features import add_v_features
from data_utils.interactions import add_interactions

def apply_feature_engineering(df, train_stats=None):
    """
    Full feature pipeline = composition of simple transformations.
    """

    df = df.copy()

    # 1. Time structure
    df = add_time_features(df)

    # 2. Amount features (with stats tracking)
    df, train_stats = add_amount_features(df, train_stats)

    # 3. V-feature normalization + anomaly signals
    df, train_stats = add_v_features(df, train_stats)

    # 4. Feature interactions (nonlinear signals)
    df = add_interactions(df)

    return df, train_stats

# --- Custom transformer for cyclic encoding ---
class CyclicalFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, cols, periods):
        self.cols = cols if isinstance(cols, list) else [cols]
        self.periods = periods if isinstance(periods, list) else [periods]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        for col, period in zip(self.cols, self.periods):
            sin_col = np.sin(2 * np.pi * X_[col] / period)
            cos_col = np.cos(2 * np.pi * X_[col] / period)
            X_[f"{col}_sin"] = sin_col
            X_[f"{col}_cos"] = cos_col
            X_.drop(columns=[col], inplace=True)  # drop original cyclic col
        return X_