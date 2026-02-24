import sys
import os
sys.path.append(os.path.abspath("src"))

import numpy as np
from preprocessing.align_fridge_h5 import load_and_align_fridge
from preprocessing.forecast_data import create_forecast_windows, save_forecast_data

LIMIT_ROWS = 1_000_000


def main():

    print("=== PREPARING MAINS FORECAST DATA ===")

    aligned = load_and_align_fridge()

    print("Limiting dataset to first 1,000,000 rows")
    aligned = aligned.head(LIMIT_ROWS)

    mains_series = aligned["power_mains"].values

    X, y = create_forecast_windows(mains_series)

    print("Forecast windows created:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    save_forecast_data(X, y, name="mains")

    print("=== DONE ===")


if __name__ == "__main__":
    main()