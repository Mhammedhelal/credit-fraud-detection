def add_interactions(df):
    """
    Interaction features = manual feature crossing.
    These capture nonlinear relationships without ML models.
    """

    df = df.copy()

    df['amount_hour_interaction'] = df['log_amount'] * df['Hour']

    df['V7_amount'] = df['V7'] * df['log_amount']
    df['V12_amount'] = df['V12'] * df['log_amount']
    df['V20_amount'] = df['V20'] * df['log_amount']

    df['V11_hour'] = df['V11'] * df['Hour']
    df['V12_hour'] = df['V12'] * df['Hour']

    return df