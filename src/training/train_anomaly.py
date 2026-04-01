import sys
import os
sys.path.append(os.path.abspath("src"))

import numpy as np
import pandas as pd
import json
import joblib

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from evaluation.metrics import (
    mean_absolute_error,
    root_mean_squared_error
)

H5_PATH = "data/ukdale/ukdale.h5"


# CNN AUTOENCODER (FAST)


def build_cnn_autoencoder(window_size):

    inputs = Input(shape=(window_size,1))

    # Encoder
    x = Conv1D(32,3,activation="relu",padding="same")(inputs)
    x = MaxPooling1D(2,padding="same")(x)

    x = Conv1D(16,3,activation="relu",padding="same")(x)
    x = MaxPooling1D(2,padding="same")(x)

    # Decoder
    x = Conv1D(16,3,activation="relu",padding="same")(x)
    x = UpSampling1D(2)(x)

    x = Conv1D(32,3,activation="relu",padding="same")(x)
    x = UpSampling1D(2)(x)

    outputs = Conv1D(1,3,activation="linear",padding="same")(x)

    model = Model(inputs,outputs)

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model




def create_sequences(data, window_size):

    sequences = []

    for i in range(len(data) - window_size):
        sequences.append(data[i:i+window_size])

    return np.array(sequences)

def main():

    print("=== TRAIN_FAST_ANOMALY_MODEL STARTED ===")

    store = pd.HDFStore(H5_PATH)
    mains = store["/building1/elec/meter1"]
    store.close()

    mains.index = pd.to_datetime(mains.index)

    if mains.index.tz:
        mains.index = mains.index.tz_localize(None)

    mains = mains.resample("6s").mean().dropna()

    series_watts = mains["power"].values.astype("float32")

    print("Loaded mains series:",series_watts.shape)


    MAX_POINTS = 200000

    if len(series_watts) > MAX_POINTS:
        series_watts = series_watts[:MAX_POINTS]

    print("Using series length:",len(series_watts))


    scaler = StandardScaler()

    series_scaled = scaler.fit_transform(series_watts.reshape(-1,1))

    os.makedirs("src/models",exist_ok=True)

    joblib.dump(scaler,"src/models/anomaly_scaler.pkl")

    print("Scaler saved")

    WINDOW_SIZE = 60

    sequences = create_sequences(series_scaled,WINDOW_SIZE)

    sequences = sequences.reshape(sequences.shape[0],WINDOW_SIZE,1)

    print("Sequence shape:",sequences.shape)

    split = int(0.8 * len(sequences))

    X_train = sequences[:split]
    X_val = sequences[split:]

    print("Train sequences:",X_train.shape)
    print("Validation sequences:",X_val.shape)

    model = build_cnn_autoencoder(WINDOW_SIZE)

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


    model.fit(
        X_train,
        X_train,
        validation_data=(X_val,X_val),
        epochs=20,
        batch_size=256,
        callbacks=[early_stop,checkpoint],
        verbose=1
    )

    print("\nEvaluating anomaly model...")

    reconstructed = model.predict(X_val,batch_size=512)

    errors = np.mean(np.square(X_val - reconstructed),axis=(1,2))

    threshold = np.percentile(errors,99)

    print("Learned anomaly threshold:",threshold)


    with open("src/models/anomaly_threshold.json","w") as f:
        json.dump({"threshold":float(threshold)},f,indent=4)

    print("Threshold saved")


    X_val_flat = X_val.reshape(-1)
    reconstructed_flat = reconstructed.reshape(-1)

    mae = mean_absolute_error(X_val_flat,reconstructed_flat)
    rmse = root_mean_squared_error(X_val_flat,reconstructed_flat)

    print("MAE:",round(mae,3))
    print("RMSE:",round(rmse,3))

    os.makedirs("metrics",exist_ok=True)

    metrics_data = {
        "model":"cnn_anomaly_autoencoder",
        "window_size":WINDOW_SIZE,
        "mae":float(mae),
        "rmse":float(rmse),
        "threshold":float(threshold)
    }

    with open("metrics/anomaly_metrics.json","w") as f:
        json.dump(metrics_data,f,indent=4)

    print("Metrics saved")

    print("=== TRAIN_FAST_ANOMALY_MODEL COMPLETED ===")


if __name__ == "__main__":
    main()