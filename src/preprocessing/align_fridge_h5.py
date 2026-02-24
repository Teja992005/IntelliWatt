import sys
import os
import pandas as pd

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import APPLIANCE_MAPPING, BUILDING, SAMPLING_RATE

H5_PATH = "data/ukdale/ukdale.h5"


def load_and_align_fridge(max_rows=1_000_000):

    store = pd.HDFStore(H5_PATH)

    mains_key = f"/building{BUILDING}/elec/meter1"
    fridge_meter = APPLIANCE_MAPPING["fridge"]
    fridge_key = f"/building{BUILDING}/elec/meter{fridge_meter}"

    mains = store[mains_key]
    fridge = store[fridge_key]

    store.close()

    mains.index = pd.to_datetime(mains.index)
    fridge.index = pd.to_datetime(fridge.index)

    if mains.index.tz:
        mains.index = mains.index.tz_localize(None)
    if fridge.index.tz:
        fridge.index = fridge.index.tz_localize(None)

    mains = mains.resample(f"{SAMPLING_RATE}s").mean()
    fridge = fridge.resample(f"{SAMPLING_RATE}s").mean()

    mains = mains.rename(columns={"power": "power_mains"})
    fridge = fridge.rename(columns={"power": "power_appliance"})

    fridge_start = fridge.index.min()
    fridge_end = fridge.index.max()

    print("Fridge time range:")
    print(fridge_start, "→", fridge_end)

    mains = mains.loc[fridge_start:fridge_end]

    aligned = pd.merge(
        mains,
        fridge,
        left_index=True,
        right_index=True,
        how="inner"
    )

    aligned = aligned.dropna()

    aligned.reset_index(inplace=True)
    aligned.rename(columns={"index": "timestamp"}, inplace=True)

    # 🔥 LIMIT DATA FOR MEMORY SAFETY
    if max_rows is not None:
        aligned = aligned.head(max_rows)
        print(f"\nLimited to first {max_rows} rows for training")

    return aligned


if __name__ == "__main__":
    df = load_and_align_fridge()
    print("Aligned fridge data shape:", df.shape)
    print(df.head())