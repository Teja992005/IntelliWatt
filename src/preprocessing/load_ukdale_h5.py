import sys
import os
import pandas as pd

# Fix import path so config works when running directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import APPLIANCE_MAPPING, BUILDING

UKDALE_PATH = "data/ukdale/ukdale.h5"


def load_building_appliance(
    appliance_name,
    aggregate_meter=1,
    max_samples=100000
):
    """
    Load mains and selected appliance using centralized config mapping.
    """

    if appliance_name not in APPLIANCE_MAPPING:
        raise ValueError(f"{appliance_name} not found in APPLIANCE_MAPPING")

    appliance_meter = APPLIANCE_MAPPING[appliance_name]

    mains_key = f"/building{BUILDING}/elec/meter{aggregate_meter}"
    app_key = f"/building{BUILDING}/elec/meter{appliance_meter}"

    # Read HDF5 tables
    mains_df = pd.read_hdf(UKDALE_PATH, mains_key).head(max_samples)
    app_df = pd.read_hdf(UKDALE_PATH, app_key).head(max_samples)

    # Extract power from MultiIndex columns
    mains_power = mains_df.xs("power", level=0, axis=1).iloc[:, 0]
    app_power = app_df.xs("power", level=0, axis=1).iloc[:, 0]

    # Convert index (timestamp) to column
    mains_clean = pd.DataFrame({
        "timestamp": mains_df.index.tz_convert(None),
        "power": mains_power.values
    })

    app_clean = pd.DataFrame({
        "timestamp": app_df.index.tz_convert(None),
        "power": app_power.values
    })

    return mains_clean, app_clean


if __name__ == "__main__":
    mains, fridge = load_building_appliance("fridge")

    print("Mains shape:", mains.shape)
    print("Fridge shape:", fridge.shape)

    print("\nMains preview:")
    print(mains.head())

    print("\nFridge preview:")
    print(fridge.head())