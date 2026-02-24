# # import joblib

# # scaler = joblib.load("src/models/nilm_kettle_scaler.pkl")
# # print("Scaler mean:", scaler.mean_)
# # print("Scaler var :", scaler.var_)
# from tensorflow.keras.models import load_model

# model = load_model("src/models/nilm_washing_machine.h5", compile=False)
# model.summary()
# import numpy as np

# y = np.load("data/processed/y_microwave.npy")

# print("Min:", y.min())
# print("Max:", y.max())
# print("Mean:", y.mean())

# print("ON samples (>800W):", np.sum(y > 800))
# print("Total samples:", len(y))
import joblib
import numpy as np

# Load your current (new) scaler
scaler = joblib.load("src/models/nilm_fridge_scaler.pkl")

print(f"Mean: {scaler.mean_}")
print(f"Scale: {scaler.scale_}")