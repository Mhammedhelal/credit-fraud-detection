def add_v_features(df, train_stats):
    """
    V-features are anonymized signals → treat them as latent variables.
    Key idea: normalize + detect anomalies.
    """
    if train_stats is None:
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
        cols = [f"V{i}_zscore" for i in range(1, 29)]
        df.drop(columns=cols, inplace=True)

        train_stats['v_stats'] = v_stats


    else:
        for i in range(1, 29):
            col = f"V{i}"
            mean = train_stats['v_stats'][col]["mean"]
            std = train_stats['v_stats'][col]["std"]
            z_col = f"{col}_zscore"
            outlier_col = f"{col}_is_outlier"
            df[z_col] = (df[col] - mean) / std
            df[outlier_col] = (df[z_col].abs() > 2).astype(int)

        cols = [f"V{i}_zscore" for i in range(1, 29)]
        df.drop(columns=cols, inplace=True)