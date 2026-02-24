import sys
import os
import pandas as pd

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import SAMPLING_RATE
from preprocessing.load_ukdale_h5 import load_building_appliance


def resample_df(df):
    """
    Resample time-series data to fixed interval using mean.
    """
    df = df.set_index("timestamp")
    df_resampled = df.resample(f"{SAMPLING_RATE}s").mean()
    df_resampled = df_resampled.dropna().reset_index()
    return df_resampled


def load_and_resample(appliance_name, max_samples=None):
    """
    Load appliance + mains and resample to config sampling rate.
    """
    mains, appliance = load_building_appliance(
        appliance_name=appliance_name,
        max_samples=max_samples
    )

    mains_rs = resample_df(mains)
    appliance_rs = resample_df(appliance)

    return mains_rs, appliance_rs


if __name__ == "__main__":
    mains, fridge = load_and_resample("fridge", max_samples=100000)

    print("After resampling:")
    print("Mains:", mains.shape)
    print("Fridge:", fridge.shape)

    print("\nMains preview:")
    print(mains.head())

    print("\nFridge preview:")
    print(fridge.head())