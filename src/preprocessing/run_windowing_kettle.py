import sys
import os

sys.path.append(os.path.abspath("src"))

import numpy as np
from preprocessing.align_kettle_h5 import load_and_align_kettle
from preprocessing.windowing import create_balanced_windows
from config import WINDOW_SIZE


def main():

    print("=== BALANCED WINDOWING: KETTLE ===")
    print(f"Using window size: {WINDOW_SIZE}")

    aligned_df = load_and_align_kettle()

    X, y = create_balanced_windows(
        mains_power=aligned_df["power_mains"].values,
        appliance_power=aligned_df["power_appliance"].values,
        appliance="kettle"
    )

    print("Final balanced dataset:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    os.makedirs("data/processed", exist_ok=True)

    np.save("data/processed/X_kettle.npy", X)
    np.save("data/processed/y_kettle.npy", y)

    print("Saved:")
    print("data/processed/X_kettle.npy")
    print("data/processed/y_kettle.npy")

    print("=== BALANCED WINDOWING COMPLETED ===")


if __name__ == "__main__":
    main()