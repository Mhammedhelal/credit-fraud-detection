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