from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def build_autoencoder(input_dim):
    """
    Dense Autoencoder for energy anomaly detection.
    """

    model = Sequential()

    # Encoder
    model.add(Dense(32, activation="relu", input_shape=(input_dim,)))
    model.add(Dense(16, activation="relu"))

    # Bottleneck
    model.add(Dense(8, activation="relu"))

    # Decoder
    model.add(Dense(16, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(input_dim, activation="linear"))

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model
