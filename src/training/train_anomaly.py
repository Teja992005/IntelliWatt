import sys
import os
sys.path.append(os.path.abspath("src"))

import numpy as np
import pandas as pd
import json
import joblib

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from models.anomaly_autoenc import build_autoencoder
from evaluation.metrics import (
    mean_absolute_error,
    root_mean_squared_error
)

H5_PATH = "data/ukdale/ukdale.h5"


def main():
    print("=== TRAIN_ANOMALY_MODEL STARTED ===")

    # --------------------------------------------------
    # Load mains power (normal behavior only)
    # --------------------------------------------------
    store = pd.HDFStore(H5_PATH)
    mains = store["/building1/elec/meter1"]
    store.close()

    mains.index = pd.to_datetime(mains.index)
    if mains.index.tz:
        mains.index = mains.index.tz_localize(None)

    mains = mains.resample("6s").mean().dropna()
    series_watts = mains["power"].values.astype("float32")

    print("Loaded mains series (watts):", series_watts.shape)

    # --------------------------------------------------
    # Subsampling (memory safety)
    # --------------------------------------------------
    MAX_POINTS = 1_000_000
    if len(series_watts) > MAX_POINTS:
        series_watts = series_watts[:MAX_POINTS]

    print("Using series length:", len(series_watts))

    # --------------------------------------------------
    # Normalize (TRAINING SPACE)
    # --------------------------------------------------
    scaler = StandardScaler()
    series_scaled = scaler.fit_transform(series_watts.reshape(-1, 1))

    os.makedirs("src/models", exist_ok=True)
    joblib.dump(scaler, "src/models/anomaly_scaler.pkl")
    print("Anomaly scaler saved")

    # --------------------------------------------------
    # Train / Validation split (time-based)
    # --------------------------------------------------
    split = int(0.8 * len(series_scaled))
    X_train = series_scaled[:split]
    X_val = series_scaled[split:]

    print("Train samples:", X_train.shape)
    print("Validation samples:", X_val.shape)

    # --------------------------------------------------
    # Build autoencoder
    # --------------------------------------------------
    model = build_autoencoder(input_dim=1)
    model.summary()

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    checkpoint = ModelCheckpoint(
        "src/models/anomaly_model.h5",
        monitor="val_loss",
        save_best_only=True
    )

    # --------------------------------------------------
    # Train
    # --------------------------------------------------
    # model.fit(
    #     X_train,
    #     X_train,
    #     validation_data=(X_val, X_val),
    #     epochs=30,
    #     batch_size=256,
    #     callbacks=[early_stop, checkpoint],
    #     verbose=1
    # )
    from tensorflow.keras.models import load_model
    model = load_model(
        "src/models/anomaly_model.h5",
         compile=False
    )
    print("Loaded trained forecasting model")
    # --------------------------------------------------
    # Evaluation (REAL WATTS)
    # --------------------------------------------------
    print("\nEvaluating ANOMALY model...")

    reconstructed_scaled = model.predict(X_val, batch_size=1024)

    # 🔥 Convert back to REAL watts
    X_val_watts = scaler.inverse_transform(X_val)
    reconstructed_watts = scaler.inverse_transform(reconstructed_scaled)

    # Per-sample anomaly score (RMSE in watts)
    errors_watts = np.sqrt(
        np.mean((X_val_watts - reconstructed_watts) ** 2, axis=1)
    )

    # Global reconstruction metrics (watts)
    mae_watts = mean_absolute_error(
        X_val_watts.flatten(),
        reconstructed_watts.flatten()
    )

    rmse_watts = root_mean_squared_error(
        X_val_watts.flatten(),
        reconstructed_watts.flatten()
    )

    print("Reconstruction MAE (Watts):", round(mae_watts, 2))
    print("Reconstruction RMSE (Watts):", round(rmse_watts, 2))

    # --------------------------------------------------
    # Threshold learning (DATA-DRIVEN, WATTS)
    # --------------------------------------------------
    threshold_watts = np.percentile(errors_watts, 99)

    print("Learned anomaly threshold (Watts):", round(threshold_watts, 2))

    with open("src/models/anomaly_threshold.json", "w") as f:
        json.dump(
            {"threshold_watts": float(threshold_watts)},
            f,
            indent=4
        )

    print("Anomaly threshold saved")

    # --------------------------------------------------
    # SAVE METRICS (REAL WATTS)
    # --------------------------------------------------
    from evaluation.save_metrics import save_metrics

    print("DEBUG → Saving metrics in WATTS")
    print("MAE:", mae_watts, "RMSE:", rmse_watts)

    save_metrics(
        appliance_name="anomaly_detection",
        mae=mae_watts,
        rmse=rmse_watts,
        f1_score=None
    )

    print("=== TRAIN_ANOMALY_MODEL COMPLETED ===")


if __name__ == "__main__":
    main()
