import pandas as pd

def load_dat(filepath, max_rows=None, skip_rows=0):
    df = pd.read_csv(
        filepath,
        sep=" ",
        header=None,
        names=["timestamp", "power"],
        nrows=max_rows,
        skiprows=skip_rows
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    return df
