import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

WINDOW_SIZE = 60
EPOCHS = 20
BATCH_SIZE = 64
MAX_SAMPLES = 200000   # 🔥 limit dataset size for faster training

def main():

    print("=== ANOMALY TRAINING STARTED ===")

    # -----------------------------------------------
    # Load mains forecast dataset
    # -----------------------------------------------
    X = np.load("data/processed/X_forecast_mains.npy")
    print("Original dataset shape:", X.shape)

    # 🔥 LIMIT DATASET
    if X.shape[0] > MAX_SAMPLES:
        X = X[:MAX_SAMPLES]
        print(f"Limited to first {MAX_SAMPLES} samples")

    # Flatten feature dimension
    X = X.reshape((X.shape[0], WINDOW_SIZE))

    # -----------------------------------------------
    # Normalize
    # -----------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = X_scaled.reshape((X_scaled.shape[0], WINDOW_SIZE, 1))

    os.makedirs("src/models", exist_ok=True)
    joblib.dump(scaler, "src/models/anomaly_scaler.pkl")
    print("Scaler saved")

    # -----------------------------------------------
    # Train / Validation split
    # -----------------------------------------------
    X_train, X_val = train_test_split(
        X_scaled,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    # -----------------------------------------------
    # Build LSTM Autoencoder
    # -----------------------------------------------
    model = Sequential()

    # Encoder
    model.add(LSTM(64, activation='relu', input_shape=(WINDOW_SIZE, 1)))
    model.add(RepeatVector(WINDOW_SIZE))

    # Decoder
    model.add(LSTM(64, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(1)))

    model.compile(optimizer='adam', loss='mse')

    model.summary()

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        X_train,
        validation_data=(X_val, X_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1
    )

    # -----------------------------------------------
    # Save Model
    # -----------------------------------------------
    model.save("src/models/anomaly_model.h5")
    print("Anomaly model saved")

    # -----------------------------------------------
    # Save Loss Curve
    # -----------------------------------------------
    os.makedirs("reports", exist_ok=True)

    plt.figure()
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction MSE")
    plt.title("Anomaly LSTM Training Curve")
    plt.legend()
    plt.savefig("reports/anomaly_loss_curve.png")
    plt.close()

    print("Saved: reports/anomaly_loss_curve.png")

    # -----------------------------------------------
    # Compute Reconstruction Error
    # -----------------------------------------------
    reconstructed = model.predict(X_val)
    reconstruction_errors = np.mean(
        np.square(X_val - reconstructed),
        axis=(1, 2)
    )

    threshold = reconstruction_errors.mean() + 2 * reconstruction_errors.std()

    print("Anomaly Threshold:", threshold)

    # -----------------------------------------------
    # Save Error Distribution Plot
    # -----------------------------------------------
    plt.figure()
    plt.hist(reconstruction_errors, bins=50)
    plt.axvline(threshold, color='r', linestyle='--', label='Threshold')
    plt.title("Reconstruction Error Distribution")
    plt.legend()
    plt.savefig("reports/anomaly_error_distribution.png")
    plt.close()

    print("Saved: reports/anomaly_error_distribution.png")

    # -----------------------------------------------
    # Save Metrics JSON
    # -----------------------------------------------
    os.makedirs("metrics", exist_ok=True)

    metrics = {
        "model": "mains_anomaly_lstm",
        "window_size": WINDOW_SIZE,
        "threshold": float(threshold),
        "mean_error": float(reconstruction_errors.mean()),
        "std_error": float(reconstruction_errors.std())
    }

    with open("metrics/anomaly_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Metrics saved to metrics/anomaly_metrics.json")

    print("=== ANOMALY TRAINING COMPLETED ===")

if __name__ == "__main__":
    main()