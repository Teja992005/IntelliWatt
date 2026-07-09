from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Dropout, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam

# seq to point cnn
def build_nilm_cnn(window_size, output_length, base_filters=16):

    model = Sequential()

    model.add(
        Conv1D(
            filters=base_filters,
            kernel_size=5,
            activation="relu",
            input_shape=(window_size, 1)
        )
    )

    model.add(
        Conv1D(
            filters=base_filters * 2,
            kernel_size=5,
            activation="relu"
        )
    )

    model.add(
        Conv1D(
            filters=base_filters * 4,
            kernel_size=3,
            activation="relu"
        )
    )

    model.add(GlobalAveragePooling1D())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(output_length))

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="mse"
    )

    return model