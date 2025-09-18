import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

def apply_feature_engineering(data, train_stats=None):
    df = data.copy()
    
    # 1. Hour + rush hour
    df['Hour'] = (df['Time'] // 3600) % 24
    df['is_rush_hour'] = df['Hour'].apply(lambda h: 1 if h in [0,1,2] else 0)

    # 2. log amount
    df['log_amount'] = np.log1p(df['Amount'])
    
    # --- if train: calculate stats ---
    if train_stats is None:
        train_stats = {}
        train_stats['log_amount_mean'] = df['log_amount'].mean()
        train_stats['log_amount_std'] = df['log_amount'].std()
        df['amount_z_scores'] = (df['log_amount'] - train_stats['log_amount_mean']) / train_stats['log_amount_std']
        df['is_outlier_amount'] = (df['amount_z_scores'].abs() > 2).astype(int)
        df.drop(columns='amount_z_scores', inplace=True)
        
        df['amount_bin'], bins = pd.qcut(
            df['log_amount'], 
            q=5, 
            labels=["Very Low", "Low", "Medium", "High", "Very High"], 
            retbins=True
        )
        train_stats['amount_bins'] = bins

        # V-features stats
        v_stats = {}
        for i in range(1, 29):
            col = f"V{i}"
            v_stats[col] = {
                "mean": df[col].mean(),
                "std": df[col].std()
            }
            z_col = f"{col}_zscore"
            outlier_col = f"{col}_is_outlier"
            df[z_col] = (df[col] - v_stats[col]["mean"]) / v_stats[col]["std"]
            df[outlier_col] = (df[z_col].abs() > 2).astype(int)
        
        # Drop unwanted
        cols = [f"V{i}_is_outlier" for i in range(1, 29) if i not in [13, 15, 22, 23, 24, 26]]
        df.drop(columns=cols, inplace=True)
        cols = [f"V{i}_zscore" for i in range(1, 29)]
        df.drop(columns=cols, inplace=True)

        train_stats['v_stats'] = v_stats

    # --- if val/test: apply stats from train ---
    else:
        mean = train_stats['log_amount_mean']
        std = train_stats['log_amount_std']
        df['amount_z_scores'] = (df['log_amount'] - mean) / std
        df['is_outlier_amount'] = (df['amount_z_scores'].abs() > 2).astype(int)
        df.drop(columns='amount_z_scores', inplace=True)

        df['amount_bin'] = pd.cut(
            df['log_amount'],
            bins=train_stats['amount_bins'],
            labels=["Very Low", "Low", "Medium", "High", "Very High"],
            include_lowest=True
        )

        for i in range(1, 29):
            col = f"V{i}"
            mean = train_stats['v_stats'][col]["mean"]
            std = train_stats['v_stats'][col]["std"]
            z_col = f"{col}_zscore"
            outlier_col = f"{col}_is_outlier"
            df[z_col] = (df[col] - mean) / std
            df[outlier_col] = (df[z_col].abs() > 2).astype(int)

        cols = [f"V{i}_is_outlier" for i in range(1, 29) if i not in [13, 15, 22, 23, 24, 26]]
        df.drop(columns=cols, inplace=True)
        cols = [f"V{i}_zscore" for i in range(1, 29)]
        df.drop(columns=cols, inplace=True)

    # 3. Interactions
    df['amount_hour_interaction'] = df['log_amount'] * df['Hour']
    df['V7_amount'] = df['V7'] * df['log_amount']
    df['V12_amount'] = df['V12'] * df['log_amount']
    df['V20_amount'] = df['V20'] * df['log_amount']
    df['V11_hour'] = df['V11'] * df['Hour']
    df['V12_hour'] = df['V12'] * df['Hour']

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