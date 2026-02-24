import sys
import os
sys.path.append(os.path.abspath("src"))

import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from models.nilm_cnn import build_nilm_cnn
from evaluation.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    power_to_state,
    f1_score
)


def main():
    print("=== TRAIN_NILM_KETTLE STARTED ===")

    # --------------------------------------------------
    # Load kettle-specific data (balanced already)
    # --------------------------------------------------
    X = np.load("data/processed/X_kettle.npy")
    y = np.load("data/processed/y_kettle.npy")

    print("Loaded kettle data")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    if X.ndim == 2:
        X = X.reshape((X.shape[0], X.shape[1], 1))

    # --------------------------------------------------
    # Train / validation split
    # --------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    # --------------------------------------------------
    # Normalize mains input only
    # --------------------------------------------------
    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train.reshape(-1, 1)
    ).reshape(X_train.shape)

    X_val = scaler.transform(
        X_val.reshape(-1, 1)
    ).reshape(X_val.shape)

    os.makedirs("src/models", exist_ok=True)
    joblib.dump(scaler, "src/models/nilm_kettle_scaler.pkl")
    print("Kettle scaler saved")

    # --------------------------------------------------
    # Build model (kettle needs stronger filters)
    # --------------------------------------------------
    model = build_nilm_cnn(
        window_size=X_train.shape[1],
        base_filters=32
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    checkpoint = ModelCheckpoint(
        "src/models/nilm_kettle.h5",
        monitor="val_loss",
        save_best_only=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )

    # --------------------------------------------------
    # Save Training Loss Curve
    # --------------------------------------------------
    os.makedirs("reports", exist_ok=True)

    plt.figure()
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Kettle Training Loss Curve")
    plt.legend()
    plt.savefig("reports/kettle_loss_curve.png")
    plt.close()

    print("Saved: reports/kettle_loss_curve.png")

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------
    print("\nEvaluating KETTLE NILM model...")

    y_val_flat = y_val.reshape(-1)
    y_pred_flat = model.predict(X_val, batch_size=1024).reshape(-1)

    mae = mean_absolute_error(y_val_flat, y_pred_flat)
    rmse = root_mean_squared_error(y_val_flat, y_pred_flat)

    print("Validation MAE (Watts):", round(mae, 2))
    print("Validation RMSE (Watts):", round(rmse, 2))

    # ON / OFF detection
    THRESHOLD = 1000

    y_true_state = power_to_state(y_val_flat, THRESHOLD)
    y_pred_state = power_to_state(y_pred_flat, THRESHOLD)

    f1 = f1_score(y_true_state, y_pred_state)
    print("ON/OFF Detection F1-score:", round(f1, 3))

    # --------------------------------------------------
    # Save Metrics Plot
    # --------------------------------------------------
    plt.figure()

    metrics_names = ["MAE", "RMSE", "F1"]
    metrics_values = [mae, rmse, f1]

    plt.bar(metrics_names, metrics_values)
    plt.title("Kettle NILM Metrics")
    plt.ylabel("Value")
    plt.savefig("reports/kettle_metrics.png")
    plt.close()

    print("Saved: reports/kettle_metrics.png")

    # --------------------------------------------------
    # Save metrics JSON
    # --------------------------------------------------
    from evaluation.save_metrics import save_metrics

    save_metrics(
        appliance_name="kettle",
        mae=mae,
        rmse=rmse,
        f1_score=f1
    )

    print("=== TRAIN_NILM_KETTLE COMPLETED ===")


if __name__ == "__main__":
    main()