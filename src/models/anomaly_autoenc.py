from tensorflow.keras import Model
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    Input,
    RepeatVector,
    TimeDistributed,
)


def build_lstm_autoencoder(window_size, latent_dim=64, dropout_rate=0.2):
    """
    Sequence-to-sequence LSTM autoencoder for mains anomaly detection.
    """

    inputs = Input(shape=(window_size, 1), name="mains_window")

    x = LSTM(128, return_sequences=True, name="encoder_lstm_1")(inputs)
    x = Dropout(dropout_rate, name="encoder_dropout_1")(x)
    x = LSTM(latent_dim, return_sequences=False, name="encoder_lstm_2")(x)
    x = Dropout(dropout_rate, name="encoder_dropout_2")(x)

    x = RepeatVector(window_size, name="repeat_latent")(x)
    x = LSTM(latent_dim, return_sequences=True, name="decoder_lstm_1")(x)
    x = Dropout(dropout_rate, name="decoder_dropout_1")(x)
    x = LSTM(128, return_sequences=True, name="decoder_lstm_2")(x)

    outputs = TimeDistributed(
        Dense(1),
        name="reconstructed_window",
    )(x)

    model = Model(inputs=inputs, outputs=outputs, name="anomaly_lstm_autoencoder")
    model.compile(optimizer="adam", loss="mse")
    return model
