import pandas as pd
import numpy as np
import os

H5_PATH = "data/ukdale/ukdale.h5"
OUT_PATH = "data/processed/mains_series.npy"

def main():
    print("Loading UK-DALE mains data...")

    store = pd.HDFStore(H5_PATH)

    # Building 1 mains (meter1)
    mains = store["/building1/elec/meter1"]
    store.close()

    # Ensure datetime index
    mains.index = pd.to_datetime(mains.index)
    if mains.index.tz:
        mains.index = mains.index.tz_localize(None)

    # Resample to 6 seconds (same as NILM)
    mains = mains.resample("6s").mean()

    # Use power column
    if "power" in mains.columns:
        series = mains["power"].values
    else:
        raise ValueError("Power column not found in mains data")

    series = series.astype("float32")
    series = series[~np.isnan(series)]

    os.makedirs("data/processed", exist_ok=True)
    np.save(OUT_PATH, series)

    print("Saved mains series")
    print("Shape:", series.shape)
    print("Min:", series.min())
    print("Max:", series.max())
    print("Mean:", series.mean())

if __name__ == "__main__":
    main()
