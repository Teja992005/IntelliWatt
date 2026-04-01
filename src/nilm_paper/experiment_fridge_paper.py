# ============================================================
# Experiment: NILM (Paper Configuration)
# Appliance: Fridge
# Dataset: UKDALE (Building 1)
# Resolution: 1 Minute
# ============================================================

import pandas as pd
import numpy as np
import time
import sys
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras import layers, models
import json
import matplotlib.pyplot as plt
RESULTS_DIR = "reports/paper_experiments"
os.makedirs(RESULTS_DIR, exist_ok=True)

BUILDING = 1
INPUT_LENGTH = 510
OUTPUT_LENGTH = 480
OVERLAP = 30
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 1e-4

H5_PATH = "data/ukdale/ukdale.h5"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import APPLIANCE_MAPPING

FRIDGE_METER = APPLIANCE_MAPPING["fridge"]

store = pd.HDFStore(H5_PATH)

mains_key = f"/building{BUILDING}/elec/meter1"
fridge_key = f"/building{BUILDING}/elec/meter{FRIDGE_METER}"

mains = store[mains_key]
fridge = store[fridge_key]

store.close()

mains.index = pd.to_datetime(mains.index)
fridge.index = pd.to_datetime(fridge.index)

if mains.index.tz:
    mains.index = mains.index.tz_localize(None)
if fridge.index.tz:
    fridge.index = fridge.index.tz_localize(None)

# Downsample to 1 minute
mains = mains.resample("1T").mean()
fridge = fridge.resample("1T").mean()

mains = mains.rename(columns={"power": "mains"})
fridge = fridge.rename(columns={"power": "fridge"})

fridge_start = fridge.index.min()
fridge_end = fridge.index.max()

mains = mains.loc[fridge_start:fridge_end]

df = pd.merge(mains, fridge, left_index=True, right_index=True, how="inner")
df = df.dropna()

mains_series = df["mains"].values
fridge_series = df["fridge"].values

print("Total samples after downsampling:", len(mains_series))


X = []
y = []

step = INPUT_LENGTH - OVERLAP

for i in range(0, len(mains_series) - INPUT_LENGTH, step):
    X.append(mains_series[i:i+INPUT_LENGTH])
    y.append(fridge_series[i:i+OUTPUT_LENGTH])

X = np.array(X)
y = np.array(y)

X = X.reshape((-1, INPUT_LENGTH, 1))

print("Number of windows:", X.shape[0])



n = len(X)
train_end = int(0.8 * n)
val_end = int(0.9 * n)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

print("Train windows:", X_train.shape[0])
print("Validation windows:", X_val.shape[0])
print("Test windows:", X_test.shape[0])

inputs = layers.Input(shape=(INPUT_LENGTH, 1))

x = layers.Conv1D(32, 3, padding="valid")(inputs)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.MaxPooling1D(pool_size=2, strides=2)(x)
b1 = layers.AveragePooling1D(pool_size=2)(x)
b2 = layers.AveragePooling1D(pool_size=4)(x)
b3 = layers.AveragePooling1D(pool_size=8)(x)
b4 = layers.AveragePooling1D(pool_size=16)(x)

b1 = layers.GlobalAveragePooling1D()(b1)
b2 = layers.GlobalAveragePooling1D()(b2)
b3 = layers.GlobalAveragePooling1D()(b3)
b4 = layers.GlobalAveragePooling1D()(b4)

x = layers.Concatenate()([b1, b2, b3, b4])

x = layers.Dense(256, activation="relu")(x)
x = layers.Dense(OUTPUT_LENGTH)(x)

model = models.Model(inputs, x)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="mse"
)

model.summary()
start_time = time.time()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

end_time = time.time()
training_time = end_time - start_time

print("Training time (seconds):", training_time)
plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.title("Paper CNN - Fridge Loss Curve")
plt.savefig(os.path.join(RESULTS_DIR, "fridge_paper_cnn_loss.png"))
plt.close()
preds = model.predict(X_test)

mae = mean_absolute_error(y_test.flatten(), preds.flatten())
rmse = np.sqrt(mean_squared_error(y_test.flatten(), preds.flatten()))

results = {
    "model": "Paper_CNN",
    "appliance": "Fridge",
    "sampling": "1_minute",
    "input_length": INPUT_LENGTH,
    "epochs": EPOCHS,
    "mae": float(mae),
    "rmse": float(rmse),
    "training_time_seconds": float(training_time)
}
MODEL_DIR = "saved_models/paper_versions"
os.makedirs(MODEL_DIR, exist_ok=True)

model_path = os.path.join(MODEL_DIR, "fridge_paper_cnn.h5")
model.save(model_path)

print("Model saved at:", model_path)
with open(os.path.join(RESULTS_DIR, "fridge_paper_cnn_metrics.json"), "w") as f:
    json.dump(results, f, indent=4)

print("\n==============================")
print("Paper CNN Results (Fridge)")
print("==============================")
print("MAE:", mae)
print("RMSE:", rmse)
print("Training Time:", training_time)
print("Results saved to:", RESULTS_DIR)