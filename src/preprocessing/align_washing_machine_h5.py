import sys
import os
import pandas as pd

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import APPLIANCE_MAPPING, BUILDING, SAMPLING_RATE

H5_PATH = "data/ukdale/ukdale.h5"


def load_and_align_washing_machine(max_rows=1_000_000):

    store = pd.HDFStore(H5_PATH)

    mains_key = f"/building{BUILDING}/elec/meter1"
    wm_meter = APPLIANCE_MAPPING["washing_machine"]
    wm_key = f"/building{BUILDING}/elec/meter{wm_meter}"

    mains = store[mains_key]
    wm = store[wm_key]

    store.close()

    mains.index = pd.to_datetime(mains.index)
    wm.index = pd.to_datetime(wm.index)

    if mains.index.tz:
        mains.index = mains.index.tz_localize(None)
    if wm.index.tz:
        wm.index = wm.index.tz_localize(None)

    mains = mains.resample(f"{SAMPLING_RATE}s").mean()
    wm = wm.resample(f"{SAMPLING_RATE}s").mean()

    mains = mains.rename(columns={"power": "power_mains"})
    wm = wm.rename(columns={"power": "power_appliance"})

    wm_start = wm.index.min()
    wm_end = wm.index.max()

    print("Washing machine time range:")
    print(wm_start, "→", wm_end)

    mains = mains.loc[wm_start:wm_end]

    aligned = pd.merge(
        mains,
        wm,
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
    df = load_and_align_washing_machine()
    print("Aligned washing machine data shape:", df.shape)
    print(df.head())