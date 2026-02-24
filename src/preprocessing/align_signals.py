def align_mains_and_appliance(mains_df, appliance_df):
    """
    Aligns mains and appliance data on timestamp.
    """

    aligned = mains_df.merge(
        appliance_df,
        on="timestamp",
        how="inner",
        suffixes=("_mains", "_appliance")
    )

    return aligned
