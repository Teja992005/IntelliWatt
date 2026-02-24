import pandas as pd
from safe_load_dat import load_dat

def force_resample(df, freq="6S"):
    df["timestamp"] = df["timestamp"].dt.floor(freq)
    return df.groupby("timestamp", as_index=False)["power"].mean()


if __name__ == "__main__":
    fridge = load_dat(
        "data/ukdale/house_1/channel_12.dat",
        max_rows=50000
    )

    # Get fridge time range
    start = fridge["timestamp"].min()
    end = fridge["timestamp"].max()

    # Load aggregate AFTER skipping early rows
    mains = load_dat(
        "data/ukdale/house_1/channel_1.dat",
        skip_rows=200000,
        max_rows=50000
    )

    # Trim aggregate to fridge time
    mains = mains[
        (mains["timestamp"] >= start) &
        (mains["timestamp"] <= end)
    ]

    print("After trimming:")
    print("Mains:", mains.shape)
    print("Fridge:", fridge.shape)

    mains_rs = force_resample(mains)
    fridge_rs = force_resample(fridge)

    aligned = pd.merge(
        mains_rs,
        fridge_rs,
        on="timestamp",
        how="inner",
        suffixes=("_mains", "_appliance")
    )

    print("\nAligned data shape:", aligned.shape)
    print(aligned.head())
