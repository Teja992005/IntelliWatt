from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def build_lstm_forecaster(window_size):
    """
    Builds an LSTM model for energy consumption forecasting.
    """

    model = Sequential([LSTM(64, activation='tanh', input_shape=(window_size, 1)), Dense(1)])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return model


if __name__ == "__main__":
    # Test model creation
    model = build_lstm_forecaster(window_size=3)
    model.summary()
