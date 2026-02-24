import sys
import os

sys.path.append(os.path.abspath("src"))

import numpy as np
from preprocessing.align_washing_machine_h5 import load_and_align_washing_machine
from preprocessing.windowing import create_balanced_windows
from config import WINDOW_SIZE


# --------------------------------------------------
# BALANCED NILM WINDOWING – WASHING MACHINE
# --------------------------------------------------

def main():

    print("=== BALANCED WINDOWING: WASHING MACHINE ===")
    print(f"Using window size: {WINDOW_SIZE}")

    aligned_df = load_and_align_washing_machine()

    X, y = create_balanced_windows(
        mains_power=aligned_df["power_mains"].values,
        appliance_power=aligned_df["power_appliance"].values,
        appliance="washing_machine"
    )

    print("Final balanced dataset:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    os.makedirs("data/processed", exist_ok=True)

    np.save("data/processed/X_washing_machine.npy", X)
    np.save("data/processed/y_washing_machine.npy", y)

    print("Saved:")
    print("data/processed/X_washing_machine.npy")
    print("data/processed/y_washing_machine.npy")

    print("=== BALANCED WINDOWING COMPLETED ===")


if __name__ == "__main__":
    main()