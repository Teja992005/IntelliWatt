import sys
import os
sys.path.append(os.path.abspath("src"))

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# --------------------------------------------------
# Visualize NILM predictions
# --------------------------------------------------

def main():

    # Load validation data
    X_val = np.load("data/processed/X_val.npy")
    y_val = np.load("data/processed/y_val.npy")

    # Load trained NILM model
    model = load_model("src/models/nilm_model.h5", compile=False)


    # Predict
    y_pred = model.predict(X_val).flatten()

    # Plot first 500 points for clarity
    N = 500

    plt.figure(figsize=(12, 4))
    plt.plot(y_val[:N], label="True Appliance Power", linewidth=2)
    plt.plot(y_pred[:N], label="Predicted Appliance Power", linestyle="--")

    plt.title("NILM – True vs Predicted Appliance Power (Validation)")
    plt.xlabel("Time Steps (6-sec intervals)")
    plt.ylabel("Power (Watts)")
    plt.legend()
    plt.grid(True)

    plt.show()

if __name__ == "__main__":
    main()
