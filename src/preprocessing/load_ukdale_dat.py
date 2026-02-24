import pandas as pd

def load_ukdale_channel(filepath, max_rows=5000):
    """
    Load a UK-DALE .dat file safely.
    Format: timestamp power
    """
    df = pd.read_csv(
        filepath,
        sep=" ",
        header=None,
        names=["timestamp", "power"],
        nrows=max_rows
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    return df


if __name__ == "__main__":
    mains_path = "data/ukdale/house_1/channel_1.dat"
    fridge_path = "data/ukdale/house_1/channel_2.dat"

    mains = load_ukdale_channel(mains_path)
    fridge = load_ukdale_channel(fridge_path)

    print("Mains sample:")
    print(mains.head())

    print("\nFridge sample:")
    print(fridge.head())
