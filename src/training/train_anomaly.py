import json
import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

sys.path.append(os.path.abspath("src"))

from models.anomaly_autoenc import build_autoencoder


WINDOW_SIZE = 60
MAX_SAMPLES = 250_000
VAL_SIZE = 0.2
EPOCHS = 30
BATCH_SIZE = 128
RANDOM_STATE = 42
DATA_PATH = "data/processed/X_forecast_mains.npy"


def load_training_windows(path, max_samples, random_state):
    """
    Load forecast mains windows and sample across the full dataset so the
    anomaly model sees a broader slice of normal behavior.
    """

    windows = np.load(path)
    windows = windows.reshape((windows.shape[0], windows.shape[1])).astype("float32")

    if len(windows) > max_samples:
        rng = np.random.default_rng(random_state)
        indices = np.sort(rng.choice(len(windows), size=max_samples, replace=False))
        windows = windows[indices]

    return windows


def compute_threshold(errors):
    """
    Use a conservative threshold from the high-error tail while still
    recording mean/std for inspection and backward compatibility.
    """

    percentile_threshold = np.percentile(errors, 99.5)
    mean_std_threshold = errors.mean() + 3 * errors.std()
    return float(max(percentile_threshold, mean_std_threshold))


def save_training_plots(history, reconstruction_errors, threshold):
    os.makedirs("reports", exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="Train Loss", linewidth=2)
    plt.plot(history.history["val_loss"], label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction MSE")
    plt.title("Anomaly Autoencoder Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/anomaly_loss_curve.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.hist(reconstruction_errors, bins=60, color="#d96c2f", alpha=0.85)
    plt.axvline(threshold, color="#117a7a", linestyle="--", linewidth=2, label="Threshold")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Frequency")
    plt.title("Anomaly Reconstruction Error Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/anomaly_error_distribution.png")
    plt.close()


def main():
    print("=== TRAIN_ANOMALY_MODEL STARTED ===")

    print(f"Loading windows from: {DATA_PATH}")
    X = load_training_windows(DATA_PATH, MAX_SAMPLES, RANDOM_STATE)
    print("Training windows shape:", X.shape)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = X_scaled.reshape((X_scaled.shape[0], WINDOW_SIZE, 1))

    os.makedirs("src/models", exist_ok=True)
    joblib.dump(scaler, "src/models/anomaly_scaler.pkl")
    print("Saved scaler: src/models/anomaly_scaler.pkl")

    X_train, X_val = train_test_split(
        X_scaled,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    print("Train shape:", X_train.shape)
    print("Validation shape:", X_val.shape)

    model = build_autoencoder(WINDOW_SIZE, latent_dim=64, dropout_rate=0.2)
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        X_train,
        validation_data=(X_val, X_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    model.save("src/models/anomaly_model.h5")
    print("Saved model: src/models/anomaly_model.h5")

    reconstructed = model.predict(X_val, batch_size=256, verbose=1)
    reconstruction_errors = np.mean(np.square(X_val - reconstructed), axis=(1, 2))
    threshold = compute_threshold(reconstruction_errors)

    save_training_plots(history, reconstruction_errors, threshold)
    print("Saved anomaly training plots")

    metrics = {
        "model": "dense_anomaly_autoencoder",
        "window_size": WINDOW_SIZE,
        "threshold": threshold,
        "mean_error": float(reconstruction_errors.mean()),
        "std_error": float(reconstruction_errors.std()),
        "validation_samples": int(len(X_val)),
        "max_samples_used": int(len(X)),
    }

    os.makedirs("metrics", exist_ok=True)
    with open("metrics/anomaly_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Saved metrics: metrics/anomaly_metrics.json")
    print("Threshold:", threshold)
    print("=== TRAIN_ANOMALY_MODEL COMPLETED ===")


if __name__ == "__main__":
    main()
