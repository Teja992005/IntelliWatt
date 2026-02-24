import numpy as np
import os

FORECAST_WINDOW = 60  # 6 minutes


def create_forecast_windows(series, window_size=FORECAST_WINDOW):
    """
    Create sequence-to-one forecasting windows.
    Input: past 60 values
    Output: next value
    """
    X = []
    y = []

    for i in range(window_size, len(series)):
        X.append(series[i - window_size:i])
        y.append(series[i])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    return X, y


def save_forecast_data(X, y, name="mains"):
    os.makedirs("data/processed", exist_ok=True)

    np.save(f"data/processed/X_forecast_{name}.npy", X)
    np.save(f"data/processed/y_forecast_{name}.npy", y)

    print("Saved:")
    print(f"data/processed/X_forecast_{name}.npy", X.shape)
    print(f"data/processed/y_forecast_{name}.npy", y.shape)


if __name__ == "__main__":
    print("Forecast data module ready.")