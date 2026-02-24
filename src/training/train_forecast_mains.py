import sys
import os
sys.path.append(os.path.abspath("src"))

import numpy as np
import joblib
import json
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error


def build_forecast_lstm(window_size):

    model = Sequential()

    model.add(LSTM(64, return_sequences=True,
                   input_shape=(window_size, 1)))
    model.add(Dropout(0.2))

    model.add(LSTM(32))
    model.add(Dropout(0.2))

    model.add(Dense(1))

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"]
    )

    return model


def main():

    print("=== TRAIN_FORECAST_MAINS STARTED ===")

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------
    X = np.load("data/processed/X_forecast_mains.npy")
    y = np.load("data/processed/y_forecast_mains.npy")

    print("Loaded forecast data")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # --------------------------------------------------
    # Train / Validation split (NO shuffle)
    # --------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False
    )

    # --------------------------------------------------
    # Normalize input only
    # --------------------------------------------------
    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train.reshape(-1, 1)
    ).reshape(X_train.shape)

    X_val = scaler.transform(
        X_val.reshape(-1, 1)
    ).reshape(X_val.shape)

    os.makedirs("src/models", exist_ok=True)
    joblib.dump(scaler, "src/models/forecast_scaler.pkl")
    print("Forecast scaler saved")

    # --------------------------------------------------
    # Build model
    # --------------------------------------------------
    model = build_forecast_lstm(window_size=X.shape[1])

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    checkpoint = ModelCheckpoint(
        "src/models/forecast_model.h5",
        monitor="val_loss",
        save_best_only=True
    )

    # --------------------------------------------------
    # Train
    # --------------------------------------------------
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=64,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )

    # --------------------------------------------------
    # Save loss curve
    # --------------------------------------------------
    os.makedirs("reports", exist_ok=True)

    plt.figure()
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Forecast Training Loss")
    plt.legend()
    plt.savefig("reports/forecast_loss_curve.png")
    plt.close()

    print("Saved: reports/forecast_loss_curve.png")

    # --------------------------------------------------
    # Evaluate
    # --------------------------------------------------
    print("\nEvaluating Forecast Model...")

    y_pred = model.predict(X_val, batch_size=1024).reshape(-1)
    y_val_flat = y_val.reshape(-1)

    mae = mean_absolute_error(y_val_flat, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val_flat, y_pred))

    print("Validation MAE:", round(mae, 3))
    print("Validation RMSE:", round(rmse, 3))

    # --------------------------------------------------
    # Save metrics graph
    # --------------------------------------------------
    plt.figure()
    plt.bar(["MAE", "RMSE"], [mae, rmse])
    plt.title("Forecast Model Metrics")
    plt.ylabel("Value")
    plt.savefig("reports/forecast_metrics.png")
    plt.close()

    print("Saved: reports/forecast_metrics.png")

    # --------------------------------------------------
    # Save metrics JSON
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

    print("=== TRAIN_FORECAST_MAINS COMPLETED ===")


if __name__ == "__main__":
    main()