import os
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import WINDOW_SIZE


# ==========================================================
# SEQ-TO-POINT SLIDING WINDOWS (CENTER-BASED)
# ==========================================================

def create_sliding_windows(
    mains_power,
    appliance_power,
    window_size=WINDOW_SIZE
):
    """
    Create sliding windows for Seq-to-Point NILM.
    Each window predicts the CENTER point.
    """

    assert len(mains_power) == len(appliance_power)

    X = []
    y = []

    half = window_size // 2

    for i in range(half, len(mains_power) - half):
        window = mains_power[i - half : i + half + 1]  # +1 for correct odd size
        target = appliance_power[i]

        if np.isnan(window).any() or np.isnan(target):
            continue

        X.append(window.reshape(-1, 1))
        y.append(target)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1, 1)


# ==========================================================
# BALANCED WINDOWING (ON/OFF BALANCING)
# ==========================================================

def create_balanced_windows(
    mains_power,
    appliance_power,
    appliance="generic",
    window_size=WINDOW_SIZE,
    on_thresholds={
        "fridge": 10,
        "kettle": 1000,
        "washing_machine": 50,
        "microwave": 800
    },
    max_off_multiplier=2
):
    """
    Memory-safe balanced windowing for NILM.
    Collects all ON windows and limited OFF windows.
    """

    threshold = on_thresholds.get(appliance, 10)

    X_on, y_on = [], []
    X_off, y_off = [], []

    total_points = len(mains_power)
    half = window_size // 2

    print("Scanning for ON windows...")

    for i in range(half, total_points - half):
        window_mains = mains_power[i - half : i + half + 1]
        target_power = appliance_power[i]

        if np.isnan(window_mains).any() or np.isnan(target_power):
            continue

        if target_power > threshold:
            X_on.append(window_mains.reshape(-1, 1))
            y_on.append(target_power)

    print(f"ON windows collected: {len(X_on)}")
    print("Sampling OFF windows...")

    max_off = len(X_on) * max_off_multiplier

    for i in range(half, total_points - half):
        if len(X_off) >= max_off:
            break

        window_mains = mains_power[i - half : i + half + 1]
        target_power = appliance_power[i]

        if np.isnan(window_mains).any() or np.isnan(target_power):
            continue

        if target_power <= threshold:
            X_off.append(window_mains.reshape(-1, 1))
            y_off.append(target_power)

    # Combine ON and OFF
    X = np.array(X_on + X_off, dtype=np.float32)
    y = np.array(y_on + y_off, dtype=np.float32).reshape(-1, 1)

    # Shuffle
    idx = np.random.permutation(len(X))
    X = X[idx]
    y = y[idx]

    return X, y


# ==========================================================
# SAVE WINDOWS
# ==========================================================

def save_nilm_windows(X, y, output_dir="data/processed"):
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "X_nilm.npy"), X)
    np.save(os.path.join(output_dir, "y_nilm.npy"), y)

    print("Saved NILM windows:")
    print("X_nilm.npy shape:", X.shape)
    print("y_nilm.npy shape:", y.shape)


# ==========================================================
# SIMPLE TEST
# ==========================================================

if __name__ == "__main__":
    mains = np.arange(2000)
    appliance = np.arange(2000) * 0.5

    X, y = create_sliding_windows(mains, appliance)

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Window size used:", WINDOW_SIZE)
    print("First window length:", len(X[0]))