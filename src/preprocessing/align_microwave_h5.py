import sys
import os
import pandas as pd

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import APPLIANCE_MAPPING, BUILDING, SAMPLING_RATE

H5_PATH = "data/ukdale/ukdale.h5"


def load_and_align_microwave(max_rows=1_000_000):

    store = pd.HDFStore(H5_PATH)

    mains_key = f"/building{BUILDING}/elec/meter1"
    microwave_meter = APPLIANCE_MAPPING["microwave"]
    microwave_key = f"/building{BUILDING}/elec/meter{microwave_meter}"

    mains = store[mains_key]
    microwave = store[microwave_key]

    store.close()

    mains.index = pd.to_datetime(mains.index)
    microwave.index = pd.to_datetime(microwave.index)

    if mains.index.tz:
        mains.index = mains.index.tz_localize(None)
    if microwave.index.tz:
        microwave.index = microwave.index.tz_localize(None)

    mains = mains.resample(f"{SAMPLING_RATE}s").mean()
    microwave = microwave.resample(f"{SAMPLING_RATE}s").mean()

    mains = mains.rename(columns={"power": "power_mains"})
    microwave = microwave.rename(columns={"power": "power_appliance"})

    microwave_start = microwave.index.min()
    microwave_end = microwave.index.max()

    print("Microwave time range:")
    print(microwave_start, "→", microwave_end)

    mains = mains.loc[microwave_start:microwave_end]

    aligned = pd.merge(
        mains,
        microwave,
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
    df = load_and_align_microwave()
    print("Aligned microwave data shape:", df.shape)
    print(df.head())