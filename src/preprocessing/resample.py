def resample_power_data(df, freq="6s"):
    """
    Resamples power data to a fixed time interval.
    """

    df = df.set_index("timestamp")

    df_resampled = df.resample(freq).mean()

    # Fill missing values with 0 (appliance OFF)
    df_resampled = df_resampled.fillna(0)

    df_resampled = df_resampled.reset_index()

    return df_resampled
