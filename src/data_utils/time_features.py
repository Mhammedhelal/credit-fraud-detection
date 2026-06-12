import numpy as np

def add_time_features(df):
    """
    Extract basic temporal structure from raw timestamp-like feature.
    """
    # Convert seconds → hour of day
    df['Hour'] = (df['Time'] // 3600) % 24

    # Insight: rush hours are NOT real rush hours here
    # (this is a weak heuristic, but keeps signal separation)
    df['is_rush_hour'] = df['Hour'].isin([0, 1, 2]).astype(int)

    return df