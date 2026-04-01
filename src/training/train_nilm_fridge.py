import sys
import os
sys.path.append(os.path.abspath("src"))

import numpy as np
import joblib
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from models.nilm_cnn import build_nilm_cnn
from evaluation.metrics import mean_absolute_error, root_mean_squared_error


def main():
    print("=== TRAIN_NILM_FRIDGE (6-sec Seq2Seq) STARTED ===")


    X = np.load("data/processed/X_fridge.npy")
    y = np.load("data/processed/y_fridge.npy")

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    if X.ndim == 2:
        X = X.reshape((X.shape[0], X.shape[1], 1))

    output_length = y.shape[1]


    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train.reshape(-1, 1)
    ).reshape(X_train.shape)

    X_val = scaler.transform(
        X_val.reshape(-1, 1)
    ).reshape(X_val.shape)

    os.makedirs("saved_models/seq2seq_6sec", exist_ok=True)
    joblib.dump(scaler, "saved_models/seq2seq_6sec/fridge_scaler.pkl")

    model = build_nilm_cnn(
        window_size=X_train.shape[1],
        output_length=output_length,
        base_filters=16
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    checkpoint = ModelCheckpoint(
        "saved_models/seq2seq_6sec/fridge_model.h5",
        monitor="val_loss",
        save_best_only=True
    )

    start_time = time.time()

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )

    training_time = time.time() - start_time

    print("Training Time:", training_time)


    os.makedirs("reports/seq2seq_6sec", exist_ok=True)

    plt.figure()
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.legend()
    plt.title("6-sec Seq2Seq Fridge Loss")
    plt.savefig("reports/seq2seq_6sec/fridge_loss.png")
    plt.close()

    y_pred = model.predict(X_val)

    mae = mean_absolute_error(y_val.flatten(), y_pred.flatten())
    rmse = root_mean_squared_error(y_val.flatten(), y_pred.flatten())

    print("Validation MAE:", mae)
    print("Validation RMSE:", rmse)

    print("=== COMPLETED ===")


if __name__ == "__main__":
    main()