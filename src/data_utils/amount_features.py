import numpy as np

def add_amount_features(df, train_stats=None):
    """
    Log transform stabilizes skewed financial distributions.
    """
    df = df.copy()

    df['log_amount'] = np.log1p(df['Amount'])

    # --- TRAIN MODE ---
    if train_stats is None:
        train_stats = {}

        mean = df['log_amount'].mean()
        std = df['log_amount'].std()

        train_stats['amount_mean'] = mean
        train_stats['amount_std'] = std

        # Z-score = anomaly signal
        z = (df['log_amount'] - mean) / std
        df['is_outlier_amount'] = (np.abs(z) > 2).astype(int)

        # Quantile binning (distribution-aware discretization)
        df['amount_bin'], bins = pd.qcut(
            df['log_amount'],
            q=5,
            labels=["Very Low", "Low", "Medium", "High", "Very High"],
            retbins=True
        )

        train_stats['amount_bins'] = bins

    # --- INFERENCE MODE ---
    else:
        mean = train_stats['amount_mean']
        std = train_stats['amount_std']

        z = (df['log_amount'] - mean) / std
        df['is_outlier_amount'] = (np.abs(z) > 2).astype(int)

        df['amount_bin'] = pd.cut(
            df['log_amount'],
            bins=train_stats['amount_bins'],
            labels=["Very Low", "Low", "Medium", "High", "Very High"],
            include_lowest=True
        )

    return df, train_stats