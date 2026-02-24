import sys
import os
sys.path.append(os.path.abspath("src"))

import numpy as np
import joblib
import json
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error


def main():

    print("=== EVALUATING SAVED FORECAST MODEL ===")

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------
    X = np.load("data/processed/X_forecast_mains.npy")
    y = np.load("data/processed/y_forecast_mains.npy")

    # Keep same split logic (no shuffle)
    split_index = int(len(X) * 0.8)

    X_val = X[split_index:]
    y_val = y[split_index:]

    print("Validation set shape:", X_val.shape)

    # --------------------------------------------------
    # Load scaler
    # --------------------------------------------------
    scaler = joblib.load("src/models/forecast_scaler.pkl")

    X_val = scaler.transform(
        X_val.reshape(-1, 1)
    ).reshape(X_val.shape)

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------
    model = load_model("src/models/forecast_model.h5", compile=False)

    # --------------------------------------------------
    # Predict
    # --------------------------------------------------
    y_pred = model.predict(X_val, batch_size=1024).reshape(-1)
    y_val_flat = y_val.reshape(-1)

    mae = mean_absolute_error(y_val_flat, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val_flat, y_pred))

    print("Validation MAE:", round(mae, 3))
    print("Validation RMSE:", round(rmse, 3))

    # --------------------------------------------------
    # Save metrics graph
    # --------------------------------------------------
    os.makedirs("reports", exist_ok=True)

    plt.figure()
    plt.bar(["MAE", "RMSE"], [mae, rmse])
    plt.title("Forecast Model Metrics")
    plt.ylabel("Value")
    plt.savefig("reports/forecast_metrics.png")
    plt.close()

    print("Saved: reports/forecast_metrics.png")

    # --------------------------------------------------
    # Save JSON
    # --------------------------------------------------
    os.makedirs("metrics", exist_ok=True)

    metrics_data = {
        "model": "mains_forecast",
        "window_size": 60,
        "mae": float(mae),
        "rmse": float(rmse)
    }

    with open("metrics/forecast_metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=4)

    print("Saved: metrics/forecast_metrics.json")

    print("=== EVALUATION COMPLETED ===")


if __name__ == "__main__":
    main()