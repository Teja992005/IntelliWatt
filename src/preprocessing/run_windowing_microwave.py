import sys
import os

# Fix import path
sys.path.append(os.path.abspath("src"))

import numpy as np
from preprocessing.align_microwave_h5 import load_and_align_microwave
from preprocessing.windowing import create_balanced_windows
from config import WINDOW_SIZE


# --------------------------------------------------
# BALANCED NILM WINDOWING – MICROWAVE
# --------------------------------------------------

def main():

    print("=== BALANCED WINDOWING: MICROWAVE ===")
    print(f"Using window size: {WINDOW_SIZE}")

    aligned_df = load_and_align_microwave()

    # ✅ USE CONFIG WINDOW SIZE (NO HARDCODING)
    X, y = create_balanced_windows(
        mains_power=aligned_df["power_mains"].values,
        appliance_power=aligned_df["power_appliance"].values,
        appliance="microwave"
    )

    print("Final balanced dataset:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # Ensure directory exists
    os.makedirs("data/processed", exist_ok=True)

    # Save appliance-specific files
    np.save("data/processed/X_microwave.npy", X)
    np.save("data/processed/y_microwave.npy", y)

    print("Saved:")
    print("data/processed/X_microwave.npy")
    print("data/processed/y_microwave.npy")

    print("=== BALANCED WINDOWING COMPLETED ===")


if __name__ == "__main__":
    main()