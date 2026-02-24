import pandas as pd

def load_power_data(filepath):
    """
    Loads electricity power data from a file.
    File format: timestamp power
    """
    df = pd.read_csv(
        filepath,
        sep=" ",
        header=None,
        names=["timestamp", "power"]
    )

    # Convert timestamp to readable datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

    return df


if __name__ == "__main__":
    print("load_data module ready")
