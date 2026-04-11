from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Input,
    Reshape,
)


def build_autoencoder(window_size, latent_dim=64, dropout_rate=0.2):
    """
    Fully-connected (Dense) autoencoder for mains anomaly detection.
    Much faster to train than the LSTM variant while retaining good
    reconstruction-error-based anomaly separation.
    """

    inputs = Input(shape=(window_size, 1), name="mains_window")

    # ── Encoder ──
    x = Reshape((window_size,), name="flatten_input")(inputs)
    x = Dense(128, activation="relu", name="encoder_dense_1")(x)
    x = Dropout(dropout_rate, name="encoder_dropout_1")(x)
    x = Dense(latent_dim, activation="relu", name="encoder_dense_2")(x)
    x = Dropout(dropout_rate, name="encoder_dropout_2")(x)

    # ── Decoder ──
    x = Dense(128, activation="relu", name="decoder_dense_1")(x)
    x = Dropout(dropout_rate, name="decoder_dropout_1")(x)
    x = Dense(window_size, activation="linear", name="decoder_dense_2")(x)

    outputs = Reshape((window_size, 1), name="reconstructed_window")(x)

    model = Model(inputs=inputs, outputs=outputs, name="anomaly_autoencoder")
    model.compile(optimizer="adam", loss="mse")
    return model
