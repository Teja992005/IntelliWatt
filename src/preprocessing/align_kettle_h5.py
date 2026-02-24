import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import APPLIANCE_MAPPING, BUILDING, SAMPLING_RATE

H5_PATH = "data/ukdale/ukdale.h5"


def load_and_align_kettle(max_rows=1_000_000):

    store = pd.HDFStore(H5_PATH)

    mains_key = f"/building{BUILDING}/elec/meter1"
    kettle_meter = APPLIANCE_MAPPING["kettle"]
    kettle_key = f"/building{BUILDING}/elec/meter{kettle_meter}"

    mains = store[mains_key]
    kettle = store[kettle_key]

    store.close()

    mains.index = pd.to_datetime(mains.index)
    kettle.index = pd.to_datetime(kettle.index)

    if mains.index.tz:
        mains.index = mains.index.tz_localize(None)
    if kettle.index.tz:
        kettle.index = kettle.index.tz_localize(None)

    mains = mains.resample(f"{SAMPLING_RATE}s").mean()
    kettle = kettle.resample(f"{SAMPLING_RATE}s").mean()

    mains = mains.rename(columns={"power": "power_mains"})
    kettle = kettle.rename(columns={"power": "power_appliance"})

    kettle_start = kettle.index.min()
    kettle_end = kettle.index.max()

    print("Kettle time range:")
    print(kettle_start, "→", kettle_end)

    mains = mains.loc[kettle_start:kettle_end]

    aligned = pd.merge(
        mains,
        kettle,
        left_index=True,
        right_index=True,
        how="inner"
    )

    aligned = aligned.dropna()

    aligned.reset_index(inplace=True)
    aligned.rename(columns={"index": "timestamp"}, inplace=True)

    # 🔥 LIMIT DATA
    if max_rows is not None:
        aligned = aligned.head(max_rows)
        print(f"\nLimited to first {max_rows} rows for training")

    return aligned


if __name__ == "__main__":
    df = load_and_align_kettle()
    print("Aligned kettle data shape:", df.shape)
    print(df.head())